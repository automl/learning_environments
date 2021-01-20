import os
import torch
import statistics
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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


def plot_hist(h1, h2, h1l, h2l, h3=None, h3l=None, xlabel=None, save_idx=None):
    plt.figure(dpi=600, figsize=(3.5, 3))
    plt.hist(h1, alpha=0.8, bins=max(1, int((max(h1)-min(h1)) / BIN_WIDTH)))

    if max(h2) == min(h2):  # if we had only a single bin
        plt.hist(h2, alpha=0.6, bins=max(1, int((max(h1)-min(h1)) / BIN_WIDTH)))
    else:
        plt.hist(h2, alpha=0.6, bins=max(1, int((max(h2)-min(h2)) / BIN_WIDTH)))

    if h3 is not None:
        plt.hist(h3, alpha=0.4, bins=max(1, int((max(h3)-min(h3)) / BIN_WIDTH)))
    plt.xlabel(xlabel)
    plt.ylabel('occurrence')
    plt.yscale('log')
    if h3 is not None:
        plt.legend((h1l, h2l, h3l), loc='upper left')
    else:
        plt.legend((h1l, h2l), loc='upper left')
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.savefig('cartpole_histogram' + str(save_idx) + '.png', bbox_inches='tight')
    plt.show()


def compare_env_output(virtual_env, replay_buffer_train_all, replay_buffer_test_all):
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

        _, reward_virtual_incorrect, _ = virtual_env.step(action=1-action, state=state)
        virt_rewards_incorrect[i] = reward_virtual_incorrect

    for i in range(len(next_states_test[0])):
        trains = next_states_train[:, i].squeeze().detach().numpy()
        tests = next_states_test[:, i].squeeze().detach().numpy()
        virts = virt_next_states[:, i].squeeze().detach().numpy()
        #diffs = diff_next_states[:,i].squeeze().detach().numpy()

        if i == 0:
            plot_name = 'cart position (next state) [m]'
        elif i == 1:
            plot_name = 'cart velocity (next state) [m/s]'
        elif i == 2:
            plot_name = 'pole angle (next state) [rad]'
        elif i == 3:
            plot_name = 'pole angular vel. (next state) [rad/s]'

        plot_hist(h1=trains,
                  h2=tests,
                  h3=virts,
                  h1l='synth. env. (train)',
                  h2l='real env. (test)',
                  h3l='synth. env. on real env. data',
                  xlabel=plot_name,
                  save_idx=i+1)

    plot_hist(h1=rewards_train.squeeze().detach().numpy(),
              h2=rewards_test.squeeze().detach().numpy(),
              h3=virt_rewards.squeeze().detach().numpy(),
              h1l='synth. env. (train)',
              h2l='real env. (test)',
              h3l='synth. env. on real env. data',
              xlabel='reward',
              save_idx=len(next_states_test[0])+1)


if __name__ == "__main__":
    # dir = '/home/dingsda/master_thesis/learning_environments/results/GTN_models_CartPole-v0'
    # file_name = 'CartPole-v0_24_I8EZDI.pt'

    dir = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_cartpole_vary_hp_2020-11-17-10/GTN_models_CartPole-v0'
    file_name = 'CartPole-v0_31_VXBIVI.pt'

    virtual_env, real_env, config = load_envs_and_config(dir=dir, file_name=file_name)
    print(config)
    config['device'] = 'cuda'
    config['agents']['ddqn']['print_rate'] = 1
    config['agents']['ddqn']['test_episodes'] = 10

    replay_buffer_train_all = ReplayBuffer(state_dim=4, action_dim=1, device='cpu')
    replay_buffer_test_all = ReplayBuffer(state_dim=4, action_dim=1, device='cpu')

    for i in range(10):
        agent = select_agent(config=config, agent_name='DDQN')
        _, _, replay_buffer_train = agent.train(env=virtual_env)
        reward, _, replay_buffer_test = agent.test(env=real_env)
        replay_buffer_train_all.merge_buffer(replay_buffer_train)
        replay_buffer_test_all.merge_buffer(replay_buffer_test)

    compare_env_output(virtual_env, replay_buffer_train_all, replay_buffer_test_all)
