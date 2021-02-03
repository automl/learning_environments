import argparse
import os

import numpy as np
import torch
import yaml

from agents.agent_utils import select_agent
from envs.env_factory import EnvFactory
from experiments.syn_env_run_vary_hp import get_all_files


def load_envs_and_config(file_name, model_dir, device):
    file_path = os.path.join(model_dir, file_name)
    save_dict = torch.load(file_path)
    config = save_dict['config']
    config['device'] = device
    env_factory = EnvFactory(config=config)
    virtual_env = env_factory.generate_virtual_env()
    virtual_env.load_state_dict(save_dict['model'])
    real_env = env_factory.generate_real_env()

    # load additional agent configs
    with open("../default_config_cartpole.yaml", "r") as stream:
        config_new = yaml.safe_load(stream)["agents"]

    config["agents"]["duelingddqn"] = config_new["duelingddqn"]
    config["agents"]["duelingddqn_vary"] = config_new["duelingddqn_vary"]

    return virtual_env, real_env, config


if __name__ == "__main__":
    model_dir = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_cartpole_vary_hp_2020-11-17-10' \
                '/GTN_models_CartPole-v0'

    # model_dir = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_acrobot_vary_hp_2020-12-12-13' \
    #             '/GTN_models_Acrobot-v1'

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, help='mode 0: real env, mode 1: syn. env. (no vary), mode 2: syn. env. (vary)', default=2)
    parser.add_argument('--model_num', type=int, help='number of models evaluated', default=40)
    parser.add_argument('--device', type=str, help='device to be used (cuda or cpu)', default='cuda')
    parser.add_argument('--agent_name', type=str, help='device to be used (cuda or cpu)', default='duelingddqn')
    parser.add_argument('--env_name', type=str, help='device to be used (cuda or cpu)', default='CartPole')
    args = parser.parse_args()

    model_num = args.model_num
    device = args.device
    mode = args.mode
    agent_name = args.agent_name
    env_name = args.env_name

    if mode == 0:
        train_on_venv = False
    elif mode == 1:
        train_on_venv = True
        with_vary_hp = False
    elif mode == 2:
        train_on_venv = True
        with_vary_hp = True

    experiment_name = "evaluate_inference_time_" + env_name + "_mode_" + str(mode) + "_" + agent_name + "_device_" + device + "_" + "time"

    train_episodes = 50
    init_episodes = 10

    time_remaining_real = 15
    time_remaining_syn = 5

    print("time_remaining_real", time_remaining_real)
    print("time_remaining_syn", time_remaining_syn)

    file_list = get_all_files(with_vary_hp=with_vary_hp, model_num=model_num, model_dir=model_dir,
                              custom_load_envs_and_config=load_envs_and_config, env_name=env_name, device=device)

    step_times_per_episode_real_env = {}
    step_times_per_episode_syn_env = {}

    file_list = file_list[:5]

    for i, file_name in enumerate(file_list):
        syn_env, real_env, config = load_envs_and_config(file_name=file_name, model_dir=model_dir, device=device)

        config["agents"][agent_name]["init_episodes"] = init_episodes
        config["agents"][agent_name]["train_episodes"] = train_episodes
        config['agents'][agent_name]['print_rate'] = 10

        print('train agents on ' + str(file_name))
        agent = select_agent(config=config, agent_name=agent_name)

        print('train on real env')
        _, _, _, step_times_real_env = agent.train(env=real_env, time_remaining=time_remaining_real)
        step_times_per_episode_real_env[real_env.env.env_name + "_" + str(i)] = {
                "step_times_per_episode_real_env": step_times_real_env,
                "step_times_mean": np.mean(np.concatenate(step_times_real_env)),
                "step_times_std": np.std(np.concatenate(step_times_real_env))
                }

        print('train on syn env')
        _, _, _, step_times_syn_env = agent.train(env=syn_env, time_remaining=time_remaining_syn)
        step_times_per_episode_syn_env[file_name] = {
                "step_times_per_episode_syn_env": step_times_syn_env,
                "step_times_mean": np.mean(np.concatenate(step_times_syn_env)),
                "step_times_std": np.std(np.concatenate(step_times_syn_env))
                }

    file_name = os.path.join(os.getcwd(), experiment_name + '.pt')
    save_dict = {}
    save_dict['config'] = config
    save_dict['file_list'] = file_list

    save_dict['step_times_per_episode_real_env'] = step_times_per_episode_real_env
    save_dict['step_times_per_episode_syn_env'] = step_times_per_episode_syn_env

    step_times_real_env_lst = [np.concatenate(v["step_times_per_episode_real_env"]) for v in step_times_per_episode_real_env.values()]
    step_times_real_env_lst = np.concatenate(step_times_real_env_lst)

    save_dict['step_times_real_env_mean'] = np.mean(step_times_real_env_lst)
    save_dict['step_times_real_env_std'] = np.std(step_times_real_env_lst)

    step_times_syn_env_lst = [np.concatenate(v["step_times_per_episode_syn_env"]) for v in step_times_per_episode_syn_env.values()]
    step_times_syn_env_lst = np.concatenate(step_times_syn_env_lst)

    save_dict['step_times_syn_env_mean'] = np.mean(step_times_syn_env_lst)
    save_dict['step_times_syn_env_std'] = np.std(step_times_syn_env_lst)

    print("device:", device)

    print("step_times_real_env_mean {:.6f}".format(save_dict['step_times_real_env_mean']))
    print("step_times_real_env_std {:.6f}".format(save_dict['step_times_real_env_std']))
    print("number of real env step calls: {}".format(step_times_real_env_lst.size))

    print("step_times_syn_env_mean {:.6f}".format(save_dict['step_times_syn_env_mean']))
    print("step_times_syn_env_std {:.6f}".format(save_dict['step_times_syn_env_std']))
    print("number of syn env step calls: {}".format(step_times_syn_env_lst.size))

    torch.save(save_dict, file_name)
