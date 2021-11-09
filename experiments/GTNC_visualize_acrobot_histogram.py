import os
import torch
import statistics
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import numpy as np
from envs.env_factory import EnvFactory
from agents.agent_utils import select_agent
from utils import ReplayBuffer

BIN_WIDTH = 0.02


def load_envs_and_config(dir, file_name):
    file_path = os.path.join(dir, file_name)
    save_dict = torch.load(file_path)
    config = save_dict['config']
    # config['envs']['CartPole-v0']['solved_reward'] = 195
    # config['envs']['CartPole-v0']['max_steps'] = 200
    env_factory = EnvFactory(config=config)
    virtual_env = env_factory.generate_virtual_env()
    virtual_env.load_state_dict(save_dict['model'])
    real_env = env_factory.generate_real_env()

    return virtual_env, real_env, config


def plot_hist(h1, h2, h1l, h2l, h3=None, h3l=None, xlabel=None, save_idx=None, agentname="none"):
    plt.figure(dpi=600, figsize=(3.5, 3))
    plt.hist(h1, alpha=0.8, bins=max(1, int((max(h1) - min(h1)) / BIN_WIDTH)))

    if max(h2) == min(h2):  # if we had only a single bin
        plt.hist(h2, alpha=0.6, bins=max(1, int((max(h1) - min(h1)) / BIN_WIDTH)))
    else:
        plt.hist(h2, alpha=0.6, bins=max(1, int((max(h2) - min(h2)) / BIN_WIDTH)))

    if h3 is not None:
        plt.hist(h3, alpha=0.4, bins=max(1, int((max(h3) - min(h3)) / BIN_WIDTH)))
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('occurrence', fontsize=12)
    plt.yscale('log')
    if h3 is not None:
        plt.legend((h1l, h2l, h3l), loc='upper left', fontsize=10)
    else:
        plt.legend((h1l, h2l), loc='upper left', fontsize=10)
    plt.subplots_adjust(bottom=0.15, left=0.16)
    plt.tick_params(labelsize=12)
    plt.savefig('acrobot_histogram_' + agentname + '_' + '223R5W' + '_' + str(save_idx) + '.png', bbox_inches='tight')
    plt.show()


def compare_env_output(virtual_env, replay_buffer_train_all, replay_buffer_test_all, agentname):
    states_train, actions_train, next_states_train, rewards_train, dones_train = replay_buffer_train_all.get_all()
    states_test, actions_test, next_states_test, rewards_test, dones_test = replay_buffer_test_all.get_all()

    virt_next_states = torch.zeros_like(next_states_test)
    virt_rewards = torch.zeros_like(rewards_test)
    virt_rewards_incorrect = torch.zeros_like(rewards_test)
    virt_dones = torch.zeros_like(dones_test)

    for i in range(len(states_test)):
        state = states_test[i]
        action = actions_test[i]

        next_state_virtual, reward_virtual, done_virtual = virtual_env.step(action=action, state=state)
        virt_next_states[i] = next_state_virtual
        virt_rewards[i] = reward_virtual
        virt_dones[i] = done_virtual

        _, reward_virtual_incorrect, _ = virtual_env.step(action=1 - action, state=state)
        virt_rewards_incorrect[i] = reward_virtual_incorrect

    for i in range(len(next_states_test[0])):
        trains = next_states_train[:, i].squeeze().detach().numpy()
        tests = next_states_test[:, i].squeeze().detach().numpy()
        virts = virt_next_states[:, i].squeeze().detach().numpy()
        # diffs = diff_next_states[:,i].squeeze().detach().numpy()

        # The state consists of the sin() and cos() of the two rotational joint
        # angles and the joint angular velocities :
        # [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2].
        if i == 0:
            plot_name = 'joint 1 [cos]'
        elif i == 1:
            plot_name = 'joint 1 [sin]'
        elif i == 2:
            plot_name = 'joint 2 [cos]'
        elif i == 3:
            plot_name = 'joint 2 [sin]'
        elif i == 4:
            plot_name = 'joint 1 [1/s]'
        elif i == 5:
            plot_name = 'joint 2 [1/s]'

        plot_hist(h1=trains,
                  h2=tests,
                  h3=virts,
                  h1l='synth. env. (train)',
                  h2l='real env. (test)',
                  h3l='synth. env. on real. env data',
                  xlabel=plot_name,
                  save_idx=i + 1,
                  agentname=agentname)

        # file_path = os.path.join(path, f"acrobot_histogram_{agentname}_'223R5W'_{i}.png")
        # plt.savefig(file_path, bbox_inches='tight', transparent=True)
        # plt.show()

    plot_hist(h1=rewards_train.squeeze().detach().numpy(),
              h2=rewards_test.squeeze().detach().numpy(),
              h3=virt_rewards.squeeze().detach().numpy(),
              h1l='synth. env. (train)',
              h2l='real env. (test)',
              h3l='synth. env. on real. env data',
              xlabel='reward',
              save_idx=len(next_states_test[0]) + 1,
              agentname=agentname)

    # file_path = os.path.join(path, f"acrobot_histogram_{agentname}_'223R5W'_{i+1}.png")
    # plt.savefig(file_path, bbox_inches='tight', transparent=True)
    # plt.show()
    # plot_diff(reals=dones.squeeze().detach().numpy(), virts=virt_dones.squeeze().detach().numpy(), diffs=diff_dones, plot_name='done')
    # plot_diff(reals=dones.squeeze().detach().numpy(), virts = virt_dones.squeeze().detach().numpy(), plot_name = 'done')


if __name__ == "__main__":
    dir = '/home/ferreira/Projects/learning_environments/results/2_thomas_results/GTNC_evaluate_acrobot_vary_hp_2020-12-12-13/GTN_models_Acrobot-v1'
    # file_name = 'Acrobot-v1_MD1EG9.pt' # used in paper for DDQN and Dueling DDQN
    file_name = 'Acrobot-v1_223R5W.pt'  # good performing dTD3

    print("using model: ", file_name)

    virtual_env, real_env, config = load_envs_and_config(dir=dir, file_name=file_name)
    print(config)

    # agentname = 'ddqn'
    agentname = 'duelingddqn'
    # agentname = 'td3_discrete_vary'

    replay_buffer_train_all = ReplayBuffer(state_dim=6, action_dim=1, device='cpu')
    replay_buffer_test_all = ReplayBuffer(state_dim=6, action_dim=1, device='cpu')

    import yaml

    with open("default_config_acrobot.yaml", "r") as stream:
        config2 = yaml.safe_load(stream)
    if 'td3' in agentname:
        config['agents'][agentname] = config2['agents']["td3_discrete_vary_layer_norm_2"]
    else:
        config['agents'][agentname] = config2['agents'][agentname]

    config['device'] = 'cuda'
    config['agents'][agentname]['print_rate'] = 1
    config['agents'][agentname]['test_episodes'] = 10

    rewards = []

    for i in range(10):
        random.seed(int(time.time()))
        np.random.seed(int(time.time()))
        torch.manual_seed(int(time.time()))
        torch.cuda.manual_seed_all(int(time.time()))
        agent = select_agent(config=config, agent_name=agentname)
        print(agent)
        _, _, replay_buffer_train = agent.train(env=virtual_env)
        reward, _, replay_buffer_test = agent.test(env=real_env)
        print(np.mean(reward))
        rewards.append(np.mean(reward))

        if "td3" in agentname:
            replay_buffer_train.make_one_hot_buffer()
            replay_buffer_test.make_one_hot_buffer()

        replay_buffer_train_all.merge_buffer(replay_buffer_train)
        replay_buffer_test_all.merge_buffer(replay_buffer_test)

    print("avg. reward", np.mean(rewards))
    compare_env_output(virtual_env, replay_buffer_train_all, replay_buffer_test_all, agentname)
