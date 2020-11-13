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
    real_env = env_factory.generate_default_real_env()

    _, _, gtn_it, _ = file_name.split('_')

    return virtual_env, real_env, config, gtn_it


def plot_hist(h1, h2, h1l, h2l, h3=None, h3l=None, xlabel=None):
    fig, ax = plt.subplots(dpi=600, figsize=(4,3))
    plt.hist(h1, alpha=0.8, bins=int((max(h1)-min(h1)) / BIN_WIDTH))
    plt.hist(h2, alpha=0.6, bins=int((max(h2)-min(h2)) / BIN_WIDTH))
    if h3 is not None:
        plt.hist(h3, alpha=0.4, bins=int((max(h3)-min(h3)) / BIN_WIDTH))
    plt.xlabel(xlabel)
    plt.ylabel('occurrence')
    plt.yscale('log')
    if h3 is not None:
        plt.legend((h1l, h2l, h3l))
    else:
        plt.legend((h1l, h2l))
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
        action = actions_train[i]
        #next_state = next_states[i]
        #reward = rewards[i]
        #done = dones[i]

        next_state_virt, reward_virtual, done_virtual = virtual_env.step(action=action, state=state)
        virt_next_states[i] = next_state_virt
        virt_rewards[i] = reward_virtual
        virt_dones[i] = done_virtual

        _, reward_virtual_incorrect, _ = virtual_env.step(action=1-action, state=state)
        virt_rewards_incorrect[i] = reward_virtual_incorrect

    # diff_next_states = virt_next_states-next_states
    # diff_rewards = virt_rewards-rewards
    # diff_dones = virt_dones-dones

    for i in range(len(next_states_test[0])):
        trains = next_states_train[:, i].squeeze().detach().numpy()
        tests = next_states_test[:, i].squeeze().detach().numpy()
        virts = virt_next_states[:, i].squeeze().detach().numpy()
        #diffs = diff_next_states[:,i].squeeze().detach().numpy()

        if i == 0:
            plot_name = 'cart position (next state)'
        elif i == 1:
            plot_name = 'cart velocity (next state)'
        elif i == 2:
            plot_name = 'pole angle (next state)'
        elif i == 3:
            plot_name = 'pole angular velocity (next state)'

        #plot_diff(reals=reals, virts=virts, diffs=diffs, plot_name=plot_name)
        plot_hist(h1=trains,
                  h2=tests,
                  h3=virts,
                  h1l='train (synth. env)',
                  h2l='test (real env)',
                  h3l='synth. env on real env data',
                  xlabel=plot_name)

    # diff_dones = diff_dones.squeeze().detach().numpy()
    # diff_rewards = diff_rewards.squeeze().detach().numpy()

    print(torch.mean(virt_rewards))
    print(torch.mean(virt_rewards_incorrect))

    plot_hist(h1=virt_rewards.squeeze().detach().numpy(),
              h2=virt_rewards_incorrect.squeeze().detach().numpy(),
              h1l='virtual environment correct action',
              h2l='virtual environment wrong action',
              xlabel='reward')
    #plot_diff(reals=dones.squeeze().detach().numpy(), virts=virt_dones.squeeze().detach().numpy(), diffs=diff_dones, plot_name='done')
    #plot_diff(reals=dones.squeeze().detach().numpy(), virts = virt_dones.squeeze().detach().numpy(), plot_name = 'done')


#def barplot_variation:
#    fill_rb_with_first_five_states_only = [200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 138.0, 142.0, 119.0, 196.0, 165.0, 150.0, 131.0, 137.0, 144.0, 123.0, 142.0, 138.0, 170.0, 163.0, 145.0, 200.0, 135.0, 200.0, 123.0, 156.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 185.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0]




def barplot_variation_hyperparameters():
    normal = [200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 194.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0]
    half_hidden_size = [200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 147.0, 200.0, 200.0, 200.0, 111.0, 141.0, 163.0, 200.0, 146.0, 145.0, 178.0, 166.0, 173.0, 146.0, 136.0, 120.0, 162.0, 200.0, 117.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0]
    double_hidden_size = [200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 199.0, 200.0, 200.0, 200.0, 193.0, 194.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0]
    half_hidden_layer = [200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 163.0, 200.0, 158.0, 200.0, 164.0, 155.0, 153.0, 138.0, 153.0, 155.0, 172.0, 176.0, 193.0, 200.0, 175.0, 184.0, 200.0, 200.0, 200.0, 185.0, 200.0, 200.0, 199.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0]
    double_hidden_layer = [200.0, 200.0, 148.0, 200.0, 158.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0]
    half_learning_rate = [200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0]
    double_learning_rate = [163.0, 167.0, 200.0, 169.0, 154.0, 200.0, 145.0, 174.0, 200.0, 175.0, 200.0, 200.0, 200.0, 145.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 183.0, 200.0, 200.0, 187.0, 200.0, 158.0, 156.0, 200.0, 200.0, 173.0, 190.0, 188.0, 185.0, 181.0, 184.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0]

    print(len(normal))

    data_dict = {'normal': normal,
                 'half hidden size': half_hidden_size,
                 'double hidden size': double_hidden_size,
                 'half hidden layer': half_hidden_layer,
                 'double hidden layer': double_hidden_layer,
                 'half learning rate': half_learning_rate,
                 'double learning rate': double_learning_rate}
    df = pd.DataFrame(data=data_dict)
    plt.subplots(dpi=600, figsize=(5,4))
    ax = sns.boxplot(data=df)
    #ax = sns.stripplot(data=df, alpha=0.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.ylabel('average cumulative reward')
    #plt.title('GTN cartpole parameter variation (10 different agents, 10 evals each agent)')
    plt.show()

if __name__ == "__main__":
    dir = '/home/dingsda/master_thesis/learning_environments/results/GTN_models__cartpole'
    file_names = ['cartpole_good_21_L5OV5M.pt', 'cartpole_good_21_HIK9XV.pt', 'cartpole_good_36_6MT1C2.pt']

    for file_name in file_names:
        virtual_env, real_env, config, gtn_it = load_envs_and_config(dir=dir, file_name=file_name)
        config['device'] = 'cuda'
        config['agents']['ddqn']['print_rate'] = 1
        config['agents']['ddqn']['test_episodes'] = 10
        #config['agents']['ddqn']['hidden_size'] = int(config['agents']['ddqn']['hidden_size']/2)
        #config['agents']['ddqn']['hidden_layer'] = int(config['agents']['ddqn']['hidden_layer']*2)
        #config['agents']['ddqn']['lr'] = config['agents']['ddqn']['lr']/2

        barplot_variation_hyperparameters()

        replay_buffer_train_all = ReplayBuffer(state_dim=4, action_dim=1, device='cpu')
        replay_buffer_test_all = ReplayBuffer(state_dim=4, action_dim=1, device='cpu')

        reward_list = []
        for i in range(2):
            print(i)
            agent = select_agent(config=config, agent_name='DDQN')
            _, replay_buffer_train = agent.train(env=virtual_env, gtn_iteration=gtn_it)
            reward, replay_buffer_test = agent.test(env=real_env)
            replay_buffer_train_all.merge_buffer(replay_buffer_train)
            replay_buffer_test_all.merge_buffer(replay_buffer_test)

            reward_list += reward

        #
        # if statistics.mean(reward) > 95:
        #     real_env.render()
        #     states, _, _, _, _ = replay_buffer_train_all.get_all()
        #     for state in states:
        #         real_env.env.env.state = state.cpu().detach().numpy()
        #         real_env.render()
        #         time.sleep(0.01)
        #
        #     states, _, _, _, _ = replay_buffer_test_all.get_all()
        #     for state in states:
        #         real_env.env.env.state = state.cpu().detach().numpy()
        #         real_env.render()
        #         time.sleep(0.01)

    print(len(reward_list))
    print(statistics.mean(reward_list))
    print(reward_list)

    compare_env_output(virtual_env, replay_buffer_train_all, replay_buffer_test_all)

    # for i in range(10):
    #     print(i)
    #     #config['agents']['ddqn']['early_out_num'] = 20
    #     #config['agents']['ddqn']['early_out_state_diff'] = 1e-6
    #     #config['agents']['ddqn']['print_rate'] = 1000
    #     agent = select_agent(config=config, agent_name='DDQN')
    #     t1 = time.time()
    #     _, replay_buffer_train = agent.train(env=virtual_env, gtn_iteration=gtn_it)
    #     print(time.time()-t1)
    #     reward, replay_buffer_test = agent.test(env=real_env)
    #     print(statistics.mean(reward))
    #     # if statistics.mean(reward) > 95:
    #     #     states, _, _, _, _ = replay_buffer_train.get_all()
    #     #     for state in states:
    #     #         print(state)
    #     #         real_env.env.env.state = state.cpu().detach().numpy()
    #     #         real_env.render()
    #     #         time.sleep(0.02)
    #         # print('entering inner loop')
    #         # _, replay_buffer = agent.test(env=virtual_env)
    #         # states, _, _, _, _ = replay_buffer.get_all()
    #         # for state in states:
    #         #     print(state)
    #         #     real_env.env.env.state = state.cpu().detach().numpy()
    #         #     real_env.render()
    #         #     time.sleep(1)



    #
    # replay_buffer_train_all = ReplayBuffer(state_dim=1, action_dim=1, device='cpu')
    # replay_buffer_test_all = ReplayBuffer(state_dim=1, action_dim=1, device='cpu')
    # q_tables = []
    # for i in range(10):
    #     print(i)
    #     agent = select_agent(config=config, agent_name='QL')
    #     _, replay_buffer_train = agent.train(env=virtual_env, gtn_iteration=gtn_it)
    #     reward, replay_buffer_test = agent.test(env=real_env)
    #     replay_buffer_train_all.merge_buffer(replay_buffer_train)
    #     replay_buffer_test_all.merge_buffer(replay_buffer_test)
    #     q_tables.append(agent.q_table)
    #
    # q_table = merge_q_tables(q_tables)
    #
    # rb_dict_train = convert_replay_buffer(replay_buffer_train_all)
    # rb_dict_test = convert_replay_buffer(replay_buffer_test_all)
    #
    # fig, ax = plt.subplots(dpi=600)
    # plot_tiles(length=0.9, n_tot=M*N)
    # plot_tile_numbers(n_tot=M*N)
    # plot_agent_behaviour(rb_dict=rb_dict_train, q_table=q_table, real_env=real_env)
    # # plot_filled_rectangle((0,1), 0.2, 0.2)
    # # plot_filled_rectangle((0,1), 0.1, 0)
    # ax.axis('equal')
    # ax.axis('off')
    # plt.savefig('gridworld_train.eps')
    #
    # fig, ax = plt.subplots(dpi=600)
    # plot_tiles(length=0.9, n_tot=M*N)
    # plot_tile_numbers(n_tot=M*N)
    # plot_agent_behaviour(rb_dict=rb_dict_test, q_table=q_table, real_env=real_env)
    # # plot_filled_rectangle((0,1), 0.2, 0.2)
    # # plot_filled_rectangle((0,1), 0.1, 0)
    # ax.axis('equal')
    # ax.axis('off')
    # plt.savefig('gridworld_test.eps')
    #
    # plt.show()