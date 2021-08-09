import argparse
import gzip
import multiprocessing as mp
import os
import pickle
import re

import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from agents.agent_utils import select_agent
from envs.env_factory import EnvFactory
from utils import save_lists

import matplotlib.pyplot as plt

MBRL_BASELINE_MODELS_PATH = "../mbrl_baseline_models"
# MBRL_PATHS = ["../mbrl_baseline", "../mbrl_baseline_1"]
MBRL_PATHS = ["../mbrl_baseline", "../mbrl_baseline_1", "../mbrl_baseline_2", "../mbrl_baseline_3", "../mbrl_baseline_4"]


def get_saved_data(mypath):
    f_names = []
    gzip_files = []
    for (dirpath, dirnames, filename) in os.walk(mypath):
        f_names.extend(filename)
        break
    for name in f_names:
        if name[-4:] == "gzip":
            gzip_files.append(name)
    return gzip_files


def load_individual_file_by_name(datasets_dir, name):
    data_file = os.path.join(datasets_dir, name)
    # print(os.path.abspath(data_file))
    # print(data_file)
    f = gzip.open(os.path.abspath(data_file), 'rb')
    data = pickle.load(f)
    return data


def get_data_for_each_bohb_run(data_filename_list):
    bohb_list_of_files = {}
    for name in data_filename_list:
        bohb_result = re.search('data_bohb_counter_(.*)_id', name)  # data_bohb_counter_0_id_4_test_counter_1
        bohb_number = bohb_result.group(1)
        id_result = re.search('id_(.*)_t', name)  # data_bohb_counter_0_id_4_test_counter_1
        id_number = id_result.group(1)
        test_counter_result = re.search('test_counter_(.*).pkl', name)  # data_bohb_counter_0_id_4_test_counter_1
        test_counter_number = test_counter_result.group(1)
        # print(f"{name}: {bohb_number} {id_number} {test_counter_number}")
        if bohb_number not in bohb_list_of_files.keys():
            bohb_list_of_files[bohb_number] = []
        bohb_list_of_files[bohb_number].append(name)
    
    return bohb_list_of_files


def data_to_xy_data(data: list):
    # print(data)
    s, a, s_n, r = [], [], [], []
    for bohb_run in data:
        for counter in bohb_run:
            for test_run in counter:
                s.append(test_run[0].numpy())
                a_one_hot = torch.nn.functional.one_hot(test_run[1], 2).squeeze().numpy()
                a.append(a_one_hot)
                s_n.append(test_run[2].numpy())
                r.append([test_run[3].numpy()])
    
    data_x = np.hstack((s, a))
    data_y = np.hstack((s_n, r))
    data_x = torch.tensor(data_x)
    data_y = torch.tensor(data_y)
    return data_x, data_y


def get_data_loader(train_x, train_y, batchsize=1024, shuffle=False):
    data_set = TensorDataset(train_x, train_y)
    train_dataloader = DataLoader(data_set, batch_size=batchsize, shuffle=shuffle)
    return train_dataloader


def load_env_wrapped(config):
    env_factory = EnvFactory(config=config)
    mbrl_baseline = env_factory.generate_virtual_env()
    real_env = env_factory.generate_real_env()
    return mbrl_baseline, real_env


def get_env_internal_models(env_wrapped):
    state_net = env_wrapped.state_net
    reward_net = env_wrapped.reward_net
    done_net = env_wrapped.done_net
    return state_net, reward_net, done_net


def fit_mbrl_baseline(mbrl_baseline, dataloader, n_epochs=50, lr=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    criterion = nn.MSELoss().to(device)
    parameters = list(mbrl_baseline.env.state_net.parameters()) + list(mbrl_baseline.env.reward_net.parameters())
    optimizer = optim.Adam(parameters, lr=lr)
    
    global_step = 0
    
    mbrl_baseline.env.train()
    loss_over_epochs = []
    for epoch in range(n_epochs):
        loss_per_epoch = []
        for step, (data, target) in enumerate(dataloader):
            global_step += step
            optimizer.zero_grad()
            states_actions, target = data.to(device).float(), target.to(device).float()
            next_states = mbrl_baseline.env.state_net(states_actions)
            rewards = mbrl_baseline.env.reward_net(states_actions)
            output = torch.hstack((next_states, rewards))
            loss = criterion(output, target)
            loss_per_epoch.append(loss.item())
            if global_step % 100 == 0:
                print(f"epoch: {epoch}, loss: {loss}")
            loss.backward()
            optimizer.step()
        loss_over_epochs.append(loss_per_epoch)
    return mbrl_baseline, loss_over_epochs

def load_baseline(file_name, config):
    save_dict = torch.load(file_name)
    env_factory = EnvFactory(config=config)
    mbrl_baseline = env_factory.generate_virtual_env()
    mbrl_baseline.load_state_dict(save_dict['model'])
    return mbrl_baseline


def  evaluate_mbrl_baseline(mbrl_baseline_models_paths, real_env, config, pool, experiment_name, agent_name, agents_num, device):
    env_reward_overview = {}
    reward_list = []
    train_steps_needed = []
    episode_length_needed = []

    config["device"] = device
    
    model_num = len(mbrl_baseline_models_paths)
    
    if pool:
        reward_list_tpl, train_steps_needed_tpl, episode_length_needed_tpl = zip(
            *pool.starmap(
                train_test_agent,
                [(load_baseline(file_name, config), real_env, config, agent_name, agents_num)
                 for file_name in mbrl_baseline_models_paths]
                )
            )

        # starmap/map preservers order of calling
        for i in range(model_num):
            reward_list += reward_list_tpl[i]
            train_steps_needed += train_steps_needed_tpl[i]
            episode_length_needed += episode_length_needed_tpl[i]
            env_reward_overview[real_env.env.env_name + "_" + str(i)] = np.hstack(reward_list_tpl[i])
    else:
        for i, file_name in enumerate(mbrl_baseline_models_paths):
            print(f'train on {file_name} environment')
            
            mbrl_baseline = load_baseline(file_name, config)
            
            reward_list_i, train_steps_needed_i, episode_length_needed_i = train_test_agent(mbrl_baseline, real_env, config, agent_name,
                                                                                            agents_num)
            
            reward_list += reward_list_i
            train_steps_needed += train_steps_needed_i
            episode_length_needed += episode_length_needed_i
            env_reward_overview[real_env.env.env_name + "_" + str(i)] = np.hstack(reward_list_i)
    
    save_lists(
        mode=3,  # mbrl baseline
        config=config,
        reward_list=reward_list,
        train_steps_needed=train_steps_needed,
        episode_length_needed=episode_length_needed,
        env_reward_overview=env_reward_overview,
        experiment_name=experiment_name
        )


def train_test_agent(train_env, test_env, config, agent_name, agents_num):
    reward_list = []
    train_steps_needed = []
    episodes_needed = []
    
    # settings for comparability
    if agent_name == "ddqn_vary":
        config['agents']['ddqn_vary']['vary_hp'] = True
        config['agents']['ddqn']['print_rate'] = 10
        config['agents']['ddqn']['early_out_num'] = 10
        config['agents']['ddqn']['train_episodes'] = 1000
        config['agents']['ddqn']['init_episodes'] = 10
        config['agents']['ddqn']['test_episodes'] = 10
        config['agents']['ddqn']['early_out_virtual_diff'] = 0.01
    
    elif agent_name == "duelingddqn_vary":
        config['agents']['duelingddqn_vary']['vary_hp'] = True
        config['agents']['duelingddqn']['print_rate'] = 10
        config['agents']['duelingddqn']['early_out_num'] = 10
        config['agents']['duelingddqn']['train_episodes'] = 1000
        config['agents']['duelingddqn']['init_episodes'] = 10
        config['agents']['duelingddqn']['test_episodes'] = 10
        config['agents']['duelingddqn']['early_out_virtual_diff'] = 0.01
    
    elif agent_name == "td3_discrete_vary":
        config['agents']['td3_discrete_vary']['vary_hp'] = True
        config['agents']['td3_discrete_vary']['print_rate'] = 10
        config['agents']['td3_discrete_vary']['early_out_num'] = 10
        config['agents']['td3_discrete_vary']['train_episodes'] = 1000
        config['agents']['td3_discrete_vary']['init_episodes'] = 10
        config['agents']['td3_discrete_vary']['test_episodes'] = 10
        config['agents']['td3_discrete_vary']['early_out_virtual_diff'] = 0.01
    else:
        raise ValueError("wrong agent_name")
    
    for i in range(agents_num):
        agent = select_agent(config=config, agent_name=agent_name)
        reward_train, episode_length, _ = agent.train(env=train_env)
        reward, _, _ = agent.test(env=test_env)
        print('reward: ' + str(reward))
        reward_list.append(reward)
        train_steps_needed.append([sum(episode_length)])
        episodes_needed.append([(len(reward_train))])
    
    return reward_list, train_steps_needed, episodes_needed


def get_raw_data(bohb_run_names):
    data_raw = []
    for k, v in bohb_run_names.items():
        data_for_bohb_individual_bohb_run = []
        for n in bohb_run_names[k]:
            d1 = load_individual_file_by_name(datasets_dir=MBRL_PATH, name=n)
            if len(d1) > 0:
                data_for_bohb_individual_bohb_run.append(d1)
        data_raw.append(data_for_bohb_individual_bohb_run)
    return data_raw


def train_supervised_model(bohb_number, data_raw, n_epochs):
    with open("../default_config_cartpole.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    
    mbrl_baseline, real_env = load_env_wrapped(config)
    
    # ATTENTION: Code for training only on bohb run 0
    data_x, data_y = data_to_xy_data(data_raw[bohb_number])
    dataloader = get_data_loader(data_x, data_y, shuffle=True)
    
    print(f"dataloader batches: {len(dataloader)}")
    # training model
    mbrl_baseline, loss_over_epochs = fit_mbrl_baseline(mbrl_baseline, dataloader, n_epochs=n_epochs)
    return mbrl_baseline, real_env, loss_over_epochs, config


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_name", type=str, help="must be in [ddqn_vary, duelingddqn_vary, td3_discrete_vary]", default="ddqn_vary")
    parser.add_argument('--pool_size', type=int, help='size of the multiprocessing pool', default=5)
    parser.add_argument('--agents_num', type=int, help='number of agents evaluated', default=20)
    parser.add_argument('--n_epochs', type=int, help='number of epochs trained', default=100)
    parser.add_argument('--device', type=str, help='device to be used (cuda or cpu)', default='gpu')
    args = parser.parse_args()
    
    agents_num = args.agents_num
    device = args.device
    agent_name = args.agent_name
    pool_size = args.pool_size
    n_epochs = args.n_epochs
    
    plot = True
    
    experiment_name = f"mbrl_baseline_model_{agent_name}_transfer_agents_num_{agents_num}_models_num_{len(MBRL_PATHS)}"

    mbrl_baseline_models_paths = []
    
    for i, MBRL_PATH in enumerate(MBRL_PATHS):
        print(f"processing train data from {MBRL_PATH}")
        print("agents_num:", agents_num, "pool size:", pool_size, "device:", device)
        
        env_name = "CartPole"
        
        # load recorded data
        data_names_list = get_saved_data(mypath=MBRL_PATH)
        bohb_run_names = get_data_for_each_bohb_run(data_filename_list=data_names_list)
        
        data_raw = get_raw_data(bohb_run_names=bohb_run_names)
        number_of_bohb_runs = len(data_raw)  # list of lists: each inner list is one bohb run
        
        # call
        mbrl_baseline, real_env, loss_over_epochs, config = train_supervised_model(bohb_number=0, data_raw=data_raw, n_epochs=n_epochs)
        
        save_dict = {}
        save_dict['model'] = mbrl_baseline.state_dict()
        save_dict['config'] = config
        save_dict['loss_over_epochs'] = loss_over_epochs
        save_path = os.path.join(MBRL_BASELINE_MODELS_PATH, f"mbrl_baseline_model_number_{i}.pt")
        print('save model: ' + str(save_path))
        torch.save(save_dict, save_path)
        mbrl_baseline_models_paths.append(save_path)
        
        if plot:
            losses = np.concatenate(loss_over_epochs)[:5000]
            plt.plot(losses)
            plt.xlabel("iteration")
            plt.ylabel("MSE")
            plt.title(f"CartPole supervised learning baseline {os.path.basename(MBRL_PATH)}")
            plt.show()
            plt.savefig(os.path.join(MBRL_BASELINE_MODELS_PATH, f"mbrl_baseline_model_number_{i}_loss.png"))
    
    if pool_size > 1:
        mp.set_start_method('spawn')  # due to cuda
        pool = mp.Pool(pool_size)
    else:
        pool = None

    device = "cpu"
    evaluate_mbrl_baseline(mbrl_baseline_models_paths, real_env, config, pool, experiment_name, agent_name, agents_num, device)
