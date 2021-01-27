import argparse
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
import torch
import yaml

from agents.agent_utils import select_agent
from envs.env_factory import EnvFactory

# local machine
# MODEL_DIR = '/home/dingsda/master_thesis/learning_environments/results/GTNC_evaluate_cartpole_vary_hp_2020-11-17-10/GTN_models_CartPole
# -v0'
# cluster
MODEL_DIR = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_cartpole_vary_hp_2020-11-17-10/GTN_models_CartPole-v0'


def load_envs_and_config(file_name):
    file_path = os.path.join(MODEL_DIR, file_name)
    save_dict = torch.load(file_path)
    config = save_dict['config']
    config['device'] = 'cuda'
    env_factory = EnvFactory(config=config)
    virtual_env = env_factory.generate_virtual_env()
    virtual_env.load_state_dict(save_dict['model'])
    real_env = env_factory.generate_real_env()

    # load additional agent configs
    print("added dueling ddqn agents in a hacky way - caution!")
    with open("../default_config_cartpole.yaml", "r") as stream:
        config_new = yaml.safe_load(stream)["agents"]

    config["agents"]["duelingddqn"] = config_new["duelingddqn"]
    config["agents"]["duelingddqn_vary"] = config_new["duelingddqn_vary"]

    return virtual_env, real_env, config


def get_all_files(with_vary_hp, model_num):
    file_list = []
    for file_name in os.listdir(MODEL_DIR):
        if 'CartPole' not in file_name:
            continue

        _, _, config = load_envs_and_config(file_name)
        if config['agents']['ddqn_vary']['vary_hp'] == with_vary_hp:
            file_list.append(file_name)

    # sort file list by random characters/digits -> make randomness deterministic
    file_list = sorted(file_list, key=lambda elem: elem[-9:])
    if len(file_list) < model_num:
        raise ValueError("Not enough saved models")

    return file_list[:model_num]


def train_test_agents(train_env, test_env, config, model_agents):
    reward_list = []
    train_steps_needed = []
    episodes_needed = []

    # settings for comparability
    config['agents']['duelingddqn_vary']['vary_hp'] = True
    config['agents']['duelingddqn']['print_rate'] = 10
    config['agents']['duelingddqn']['early_out_num'] = 10
    config['agents']['duelingddqn']['train_episodes'] = 1000
    config['agents']['duelingddqn']['init_episodes'] = 10
    config['agents']['duelingddqn']['test_episodes'] = 10
    config['agents']['duelingddqn']['early_out_virtual_diff'] = 0.01

    for i in range(model_agents):
        agent = select_agent(config=config, agent_name='DuelingDDQN_vary')
        reward_train, episode_length, _ = agent.train(env=train_env)
        reward, _, _ = agent.test(env=test_env)
        print('reward: ' + str(reward))
        reward_list.append(reward)
        train_steps_needed.append([sum(episode_length)])
        episodes_needed.append([(len(reward_train))])

    return reward_list, train_steps_needed, episodes_needed


def save_lists(mode, config, reward_list, train_steps_needed, episode_length_needed, env_reward_overview, experiment_name=None):
    file_name = os.path.join(os.getcwd(), str(mode) + '_' + experiment_name + '.pt')
    save_dict = {}
    save_dict['config'] = config
    save_dict['reward_list'] = reward_list
    save_dict['train_steps_needed'] = train_steps_needed
    save_dict['episode_length_needed'] = episode_length_needed
    save_dict['env_reward_overview'] = pd.DataFrame.from_dict(env_reward_overview, orient="index")

    torch.save(save_dict, file_name)


def run_vary_hp(mode, experiment_name, model_num, model_agents, pool=None):
    if mode == 0:
        train_on_venv = False
    elif mode == 1:
        train_on_venv = True
        with_vary_hp = False
    elif mode == 2:
        train_on_venv = True
        with_vary_hp = True

    env_reward_overview = {}
    reward_list = []
    train_steps_needed = []
    episode_length_needed = []

    if not train_on_venv:
        file_name = os.listdir(MODEL_DIR)[0]
        _, real_env, config = load_envs_and_config(file_name)

        if pool is None:
            for i in range(model_num):
                print('train on {}-th environment'.format(i))
                reward_list_i, train_steps_needed_i, episode_length_needed_i = train_test_agents(train_env=real_env, test_env=real_env,
                                                                                                 config=config, model_agents=model_agents)
                reward_list += reward_list_i
                train_steps_needed += train_steps_needed_i
                episode_length_needed += episode_length_needed_i
                env_reward_overview[real_env.env.env_name + "_" + str(i)] = np.hstack(reward_list_i)
        else:
            reward_list_tpl, train_steps_needed_tpl, episode_length_needed_tpl = zip(*pool.starmap(train_test_agents,
                                                                                                   [(real_env, real_env, config,
                                                                                                     model_agents)
                                                                                                    for _ in range(model_num)])
                                                                                     )

            # starmap/map preservers order of calling
            for i in range(model_num):
                reward_list += reward_list_tpl[i]
                train_steps_needed += train_steps_needed_tpl[i]
                episode_length_needed += episode_length_needed_tpl[i]
                env_reward_overview[real_env.env.env_name + "_" + str(i)] = np.hstack(reward_list_tpl[i])

    else:
        file_list = get_all_files(with_vary_hp=with_vary_hp, model_num=model_num)

        if pool is None:
            for file_name in file_list:
                virtual_env, real_env, config = load_envs_and_config(file_name)
                print('train agents on ' + str(file_name))

                reward_list_i, train_steps_needed_i, episode_length_needed_i = train_test_agents(train_env=virtual_env, test_env=real_env,
                                                                                                 config=config, model_agents=model_agents)
                reward_list += reward_list_i
                train_steps_needed += train_steps_needed_i
                episode_length_needed += episode_length_needed_i
                env_reward_overview[file_name] = np.hstack(reward_list_i)
        else:
            _, _, config = load_envs_and_config(file_list[0])

            reward_list_tpl, train_steps_needed_tpl, episode_length_needed_tpl = zip(*pool.starmap(train_test_agents,
                                                                                                   [(*load_envs_and_config(file_name),
                                                                                                     model_agents)
                                                                                                    for file_name in file_list]
                                                                                                   )
                                                                                     )
            # starmap/map preservers order of calling
            for i, file_name in enumerate(file_list):
                reward_list += reward_list_tpl[i]
                train_steps_needed += train_steps_needed_tpl[i]
                episode_length_needed += episode_length_needed_tpl[i]
                env_reward_overview[file_name] = np.hstack(reward_list_tpl[i])

    save_lists(mode=mode,
               config=config,
               reward_list=reward_list,
               train_steps_needed=train_steps_needed,
               episode_length_needed=episode_length_needed,
               env_reward_overview=env_reward_overview,
               experiment_name=experiment_name
               )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, help='mode 0: real env, mode 1: syn. env. (no vary), mode 2: syn. env. (vary)')
    parser.add_argument('--pool', type=int, help='size of the multiprocessing pool')
    parser.add_argument('--model_agents', type=int, help='number of agents evaluated', default=10)
    parser.add_argument('--model_num', type=int, help='number of models evaluated', default=40)
    args = parser.parse_args()

    model_num = args.model_num
    model_agents = args.model_agents

    print("model_num: ", model_num, "model_agents: ", model_agents, "pool size: ", args.pool)
    experiment_name = "ddqn_to_duelingddqn_vary_transfer_episode_steps_" + \
                      str(model_agents) + "_model_agents_" + str(model_num) + "_model_num"

    if args.pool is not None:
        mp.set_start_method('spawn')  # due to cuda
        pool = mp.Pool(args.pool)
    else:
        pool = None

    if args.mode is not None:
        run_vary_hp(mode=args.mode, experiment_name=experiment_name, model_num=model_num, model_agents=model_agents, pool=pool)
    else:
        for mode in range(3):
            run_vary_hp(mode=mode, experiment_name=experiment_name, model_num=model_num, model_agents=model_agents, pool=pool)
