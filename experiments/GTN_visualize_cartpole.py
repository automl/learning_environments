import os
import torch
import statistics
import time
import matplotlib.pyplot as plt
from envs.env_factory import EnvFactory
from agents.agent_utils import select_agent
from utils import ReplayBuffer


def load_envs_and_config(dir, file_name):
    file_path = os.path.join(dir, file_name)
    save_dict = torch.load(file_path)
    config = save_dict['config']
    config['agents']['ddqn']['rb_size'] = 100000
    # config['envs']['CartPole-v0']['solved_reward'] = 195
    # config['envs']['CartPole-v0']['max_steps'] = 200
    env_factory = EnvFactory(config=config)
    virtual_env = env_factory.generate_virtual_env()
    virtual_env.load_state_dict(save_dict['model'])
    real_env = env_factory.generate_default_real_env()

    _, _, gtn_it, _ = file_name.split('_')

    return virtual_env, real_env, config, gtn_it


def plot_diff(reals, virts, diffs, plot_name):
    fig, ax = plt.subplots(dpi=600)
    plt.hist(reals, 50, alpha=0.8)
    plt.hist(virts, 50, alpha=0.6)
    #plt.hist(diffs, 50, alpha=0.4)
    plt.xlabel(plot_name)
    plt.ylabel('occurrence')
    plt.legend(('real', 'virtual', 'difference'))
    plt.show()


def compare_env_output(virtual_env, replay_buffer):
    states, actions, next_states, rewards, dones = replay_buffer.get_all()

    virt_next_states = torch.zeros_like(states)
    virt_rewards = torch.zeros_like(rewards)
    virt_dones = torch.zeros_like(dones)

    for i in range(len(states)):
        state = states[i]
        action = actions[i]
        next_state = next_states[i]
        reward = rewards[i]
        done = dones[i]

        next_state_virt, reward_virtual, done_virtual = virtual_env.step(action=action, state=state)
        virt_next_states[i] = next_state_virt-next_state
        virt_rewards[i] = reward_virtual-reward
        virt_dones[i] = done_virtual-done

    diff_next_states = virt_next_states-next_states
    diff_rewards = virt_rewards-rewards
    diff_dones = virt_dones-dones

    for i in range(len(diff_next_states[0])):
        reals = next_states[:, i].squeeze().detach().numpy()
        virts = virt_next_states[:, i].squeeze().detach().numpy()
        diffs = diff_next_states[:,i].squeeze().detach().numpy()

        if i == 0:
            plot_name = 'cart position'
        elif i == 1:
            plot_name = 'cart velocity'
        elif i == 2:
            plot_name = 'pole angle'
        elif i == 3:
            plot_name = 'pole angular velocity'

        plot_diff(reals=reals, virts=virts, diffs=diffs, plot_name=plot_name)

    diff_dones = diff_dones.squeeze().detach().numpy()
    diff_rewards = diff_rewards.squeeze().detach().numpy()

    plot_diff(reals=rewards, virts=virts, diffs=diff_rewards, plot_name='reward')
    plot_diff(reals=dones, virts=virts, diffs=diff_dones, plot_name='done')


if __name__ == "__main__":
    dir = '/home/dingsda/master_thesis/learning_environments/results/GTN_models_cartpole'
    file_name = 'cartpole_good_21_HIK9XV.pt'
    virtual_env, real_env, config, gtn_it = load_envs_and_config(dir=dir, file_name=file_name)
    config['device'] = 'cuda'
    config['agents']['ddqn']['print_rate'] = 1

    replay_buffer_train_all = ReplayBuffer(state_dim=4, action_dim=1, device='cpu')
    replay_buffer_test_all = ReplayBuffer(state_dim=4, action_dim=1, device='cpu')

    for i in range(3):
        print(i)
        agent = select_agent(config=config, agent_name='DDQN')
        _, replay_buffer_train = agent.train(env=virtual_env, gtn_iteration=gtn_it)
        reward, replay_buffer_test = agent.test(env=real_env)
        replay_buffer_train_all.merge_buffer(replay_buffer_train)
        replay_buffer_test_all.merge_buffer(replay_buffer_test)

        if statistics.mean(reward) > 95:
            real_env.render()
            time.sleep(10)
            states, _, _, _, _ = replay_buffer_train_all.get_all()
            for state in states:
                real_env.env.env.state = state.cpu().detach().numpy()
                real_env.render()
                time.sleep(0.02)

            time.sleep(5)
            states, _, _, _, _ = replay_buffer_test_all.get_all()
            for state in states:
                real_env.env.env.state = state.cpu().detach().numpy()
                real_env.render()
                time.sleep(0.02)

    #diff_states, diff_rewards, diff_dones = compare_env_output(virtual_env, replay_buffer_train_all)

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