import argparse
import multiprocessing as mp
import os

import torch
import yaml

from agents.agent_utils import select_agent
from envs.env_factory import EnvFactory
from experiments.syn_env_run_vary_hp import run_vary_hp


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
    with open("default_config_cartpole.yaml", "r") as stream:
        config_new = yaml.safe_load(stream)["agents"]

    config["agents"]["duelingddqn"] = config_new["duelingddqn"]
    config["agents"]["duelingddqn_vary"] = config_new["duelingddqn_vary"]

    return virtual_env, real_env, config


def train_test_agents(train_env, test_env, config, agents_num):
    reward_list_synthetic = []
    reward_list_real = []
    
    train_steps_needed_synthetic = []
    episodes_needed_synthetic = []
    
    train_steps_needed_real = []
    episodes_needed_real = []
    
    # settings for comparability
    config['agents']['duelingddqn_vary']['vary_hp'] = True
    config['agents']['duelingddqn']['print_rate'] = 10
    config['agents']['duelingddqn']['early_out_num'] = 10
    config['agents']['duelingddqn']['train_episodes'] = 1000
    config['agents']['duelingddqn']['init_episodes'] = 10
    config['agents']['duelingddqn']['test_episodes'] = 100
    config['agents']['duelingddqn']['early_out_virtual_diff'] = 0.01
    
    for i in range(agents_num):
        # synthetic
        agent = select_agent(config=config, agent_name='DuelingDDQN_vary')
        config_varied = agent.full_config
        for j in range(100):
            agent = select_agent(config=config_varied, agent_name='DuelingDDQN')
            print(f"training on syn env with config (agent {j}): {agent.full_config}")
            reward_train, episode_length, _ = agent.train(env=train_env)
            reward, _, _ = agent.test(env=test_env)
            print('reward when trained on synth. env: ' + str(reward))
            reward_list_synthetic.append(reward)
            train_steps_needed_synthetic.append([sum(episode_length)])
            episodes_needed_synthetic.append([(len(reward_train))])
            
            # real
            agent = select_agent(config=config_varied, agent_name='DuelingDDQN')
            print(f"training on real env with config (agent {j}): {agent.full_config}")
            reward_train, episode_length, _ = agent.train(env=test_env)
            reward, _, _ = agent.test(env=test_env)
            print('reward when trained on real env: ' + str(reward))
            reward_list_real.append(reward)
            train_steps_needed_real.append([sum(episode_length)])
            episodes_needed_real.append([(len(reward_train))])
    
    reward_dct = {
        "config":    config_varied,
        "synthetic": reward_list_synthetic,
        "real":      reward_list_real
        }
    
    train_steps_dct = {
        "config":    config_varied,
        "synthetic": train_steps_needed_synthetic,
        "real":      train_steps_needed_real
        }
    
    episodes_needed_dct = {
        "config":    config_varied,
        "synthetic": episodes_needed_synthetic,
        "real":      episodes_needed_real
        }
    
    return reward_dct, train_steps_dct, episodes_needed_dct


if __name__ == "__main__":
    model_dir = '/home/ferreira/Projects/learning_environments/results/2_thomas_results/GTNC_evaluate_cartpole_vary_hp_2020-11-17-10/GTN_models_CartPole-v0'
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--mode', type=int, help='mode 0: real env, mode 1: syn. env. (no vary), mode 2: syn. env. (vary)')
    parser.add_argument('--pool', type=int, help='size of the multiprocessing pool')
    parser.add_argument('--agents_num', type=int, help='number of agents evaluated', default=10)
    parser.add_argument('--model_num', type=int, help='number of models evaluated', default=40)
    parser.add_argument('--device', type=str, help='device to be used (cuda or cpu)', default='cuda')
    args = parser.parse_args()
    
    args.mode = 2
    model_num = args.model_num
    agents_num = args.agents_num
    device = args.device
    
    print("model_num:", model_num, "agents_num:", agents_num, "pool size:", args.pool, "device:", device)
    
    experiment_name = "duelingddqn_vary_correlation_syn_real_" + str(agents_num) + "_agents_num_" + str(model_num) + "_model_num"
    
    env_name = "CartPole"
    
    if args.pool is not None:
        mp.set_start_method('spawn')  # due to cuda
        pool = mp.Pool(args.pool)
    else:
        pool = None
    
    if args.mode is not None:
        run_vary_hp(
            mode=args.mode, experiment_name=experiment_name, model_num=model_num, agents_num=agents_num,
            model_dir=model_dir, custom_load_envs_and_config=load_envs_and_config,
            custom_train_test_agents=train_test_agents, env_name=env_name, pool=pool, device=device, correlation_exp=True,
            )
    else:
        for mode in range(3):
            run_vary_hp(
                mode=mode, experiment_name=experiment_name, model_num=model_num, agents_num=agents_num,
                model_dir=model_dir, custom_load_envs_and_config=load_envs_and_config,
                custom_train_test_agents=train_test_agents, env_name=env_name, pool=pool, device=device, correlation_exp=True,
                )
