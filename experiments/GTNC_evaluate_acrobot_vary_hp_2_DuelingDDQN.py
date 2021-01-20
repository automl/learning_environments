import os
import sys

import torch
import yaml

from agents.agent_utils import select_agent
from envs.env_factory import EnvFactory

MODEL_NUM = 40
MODEL_AGENTS = 10
# local machine
# MODEL_DIR = '/home/dingsda/master_thesis/learning_environments/results/GTNC_evaluate_cartpole_vary_hp_2020-11-17-10/GTN_models_CartPole
# -v0'
# cluster
MODEL_DIR = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_acrobot_vary_hp_2020-12-12-13/GTN_models_Acrobot-v1'


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
    with open("../default_config_acrobot.yaml", "r") as stream:
        config_new = yaml.safe_load(stream)["agents"]

    config["agents"]["duelingddqn"] = config_new["duelingddqn"]
    config["agents"]["duelingddqn_vary"] = config_new["duelingddqn_vary"]

    return virtual_env, real_env, config


def get_all_files(with_vary_hp):
    file_list = []
    for file_name in os.listdir(MODEL_DIR):
        if 'Acrobot' not in file_name:
            continue

        _, _, config = load_envs_and_config(file_name)
        if config['agents']['ddqn_vary']['vary_hp'] == with_vary_hp:
            file_list.append(file_name)

    # sort file list by random characters/digits -> make randomness deterministic
    file_list = sorted(file_list, key=lambda elem: elem[-9:])
    if len(file_list) < MODEL_NUM:
        raise ValueError("Not enough saved models")

    return file_list[:MODEL_NUM]


def train_test_agents(train_env, test_env, config):
    reward_list = []
    train_steps_needed = []

    # settings for comparability
    config['agents']['duelingddqn_vary']['vary_hp'] = True
    config['agents']['duelingddqn']['print_rate'] = 10
    config['agents']['duelingddqn']['early_out_num'] = 10
    config['agents']['duelingddqn']['train_episodes'] = 1000
    config['agents']['duelingddqn']['init_episodes'] = 10
    config['agents']['duelingddqn']['test_episodes'] = 10
    config['agents']['duelingddqn']['early_out_virtual_diff'] = 0.01

    config["agents"]["ddqn"]["early_out_episode"] = 1

    for i in range(MODEL_AGENTS):
        agent = select_agent(config=config, agent_name='DuelingDDQN_vary')
        reward_train, _, _ = agent.train(env=train_env)
        reward, _, _ = agent.test(env=test_env)
        print('reward: ' + str(reward))
        reward_list.append(reward)
        train_steps_needed.append([(len(reward_train))])

    return reward_list, train_steps_needed


def save_lists(mode, config, reward_list, train_steps_needed, experiment_name=None):
    file_name = os.path.join(os.getcwd(), str(mode) + '_' + experiment_name + '.pt')
    save_dict = {}
    save_dict['config'] = config
    save_dict['reward_list'] = reward_list
    save_dict['train_steps_needed'] = train_steps_needed
    torch.save(save_dict, file_name)


def run_vary_hp(mode, experiment_name):
    if mode == 0:
        train_on_venv = False
    elif mode == 1:
        train_on_venv = True
        with_vary_hp = False
    elif mode == 2:
        train_on_venv = True
        with_vary_hp = True

    reward_list = []
    train_steps_needed = []

    if not train_on_venv:
        file_name = os.listdir(MODEL_DIR)[0]
        _, real_env, config = load_envs_and_config(file_name)

        for i in range(MODEL_NUM):  # 40
            print('train on {}-th environment'.format(i))
            reward_list_i, train_steps_needed_i = train_test_agents(train_env=real_env, test_env=real_env, config=config)
            reward_list += reward_list_i
            train_steps_needed += train_steps_needed_i

    else:
        file_list = get_all_files(with_vary_hp=with_vary_hp)

        for file_name in file_list:
            virtual_env, real_env, config = load_envs_and_config(file_name)
            print('train agents on ' + str(file_name))
            reward_list_i, train_steps_needed_i = train_test_agents(train_env=virtual_env, test_env=real_env, config=config)
            reward_list += reward_list_i
            train_steps_needed += train_steps_needed_i

    save_lists(mode=mode, config=config, reward_list=reward_list, train_steps_needed=train_steps_needed, experiment_name=experiment_name)


if __name__ == "__main__":
    experiment_name = "ddqn_to_duelingddqn_vary_transfer_acrobot_episode_length"
    if len(sys.argv) > 1:
        run_vary_hp(mode=int(int(sys.argv[1])), experiment_name=experiment_name)
    else:
        for mode in range(3):
            run_vary_hp(mode=mode, experiment_name=experiment_name)
