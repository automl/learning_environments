import matplotlib.pyplot as plt
import os
from envs.env_factory import EnvFactory
from agents.agent_utils import select_agent
from utils import ReplayBuffer
import torch
import numpy as np
from collections import defaultdict

# from gridworld.py
G_RIGHT = 0
G_LEFT = 1
G_DOWN = 2
G_UP = 3

FONTSIZE_LARGE = 14
FONTSIZE_SMALL = 10

NS_OFFSET = 0.3
NS_DIFF = 0.15

COLOR_R = '1f'
COLOR_G = '77'
COLOR_B = 'b4'
COLOR = '#' + COLOR_R + COLOR_G + COLOR_B


def idx_to_xy(idx):
    n = N
    x = idx // n
    y = idx % n
    return y, -x


def xy_to_idx(xy):
    n = N
    y, x = xy
    obs = -x * n + y
    return obs


def map_intensity_to_color(intensity):
    base_r = int(COLOR_R, 16)
    base_g = int(COLOR_G, 16)
    base_b = int(COLOR_B, 16)

    inter_r = hex(int(255.0 + (base_r-255.0)*intensity))[2:]
    inter_g = hex(int(255.0 + (base_g-255.0)*intensity))[2:]
    inter_b = hex(int(255.0 + (base_b-255.0)*intensity))[2:]

    return '#' + inter_r + inter_g + inter_b


def plot_tiles(length, n_tot):
    for idx in range(n_tot):
        center = idx_to_xy(idx)
        x1 = center[0]-length/2
        x2 = center[0]+length/2
        y1 = center[1]-length/2
        y2 = center[1]+length/2
        plt.plot([x1,x1, x2, x2, x1], [y1, y2, y2, y1, y1], color=COLOR, linewidth=1)


def plot_filled_rectangle(center, length, intensity):
    x = center[0] - length/2
    y = center[1] - length/2
    rect = plt.Rectangle((x, y), length, length, facecolor=map_intensity_to_color(intensity), fill=True)
    plt.gca().add_patch(rect)


def plot_filled_circle(center, diameter, intensity):
    rect = plt.Circle(center, diameter/2, facecolor=map_intensity_to_color(intensity), fill=True)
    plt.gca().add_patch(rect)


def plot_tile_numbers(n_tot):
    for idx in range(n_tot):
        center = idx_to_xy(idx)
        text = str(idx)
        if idx == 0:
            text = str(idx)+'(S)'
        elif idx == n_tot-1:
            text = str(idx) + '(G)'
        plt.gca().text(center[0], center[1], text, fontsize=FONTSIZE_LARGE, fontweight='bold', ha='center', va='center')


def load_envs_and_config(dir, file_name):
    file_path = os.path.join(dir, file_name)
    save_dict = torch.load(file_path)
    config = save_dict['config']

    env_factory = EnvFactory(config=config)
    virtual_env = env_factory.generate_virtual_env()
    virtual_env.load_state_dict(save_dict['model'])
    real_env = env_factory.generate_default_real_env()

    grid_size, gtn_it, _ = file_name.split('_')
    M, N = grid_size.split('x')
    return virtual_env, real_env, config, int(M), int(N), int(gtn_it)


def convert_to_int_list(tensor_list):
    return [elem.item() for elem in tensor_list.int()]


def convert_to_float_list(tensor_list):
    return [elem.item() for elem in tensor_list.float()]


def convert_replay_buffer(replay_buffer):
    rb_dict = defaultdict(dict)

    states, actions, next_states, rewards, dones = replay_buffer.get_all()
    print(states)
    states = convert_to_int_list(states)
    actions = convert_to_int_list(actions)
    next_states = convert_to_int_list(next_states)
    rewards = convert_to_float_list(rewards)

    for state, action, next_state, reward in zip(states, actions, next_states, rewards):
        if action not in rb_dict[state]:
            rb_dict[state][action] = {'next_states': [next_state], 'rewards': [reward]}
        else:
            rb_dict[state][action]['next_states'].append(next_state)
            rb_dict[state][action]['rewards'].append(reward)

    for state, state_dict in rb_dict.items():
        for action, action_dict in state_dict.items():
            action_dict['avg_rewards'] = {}
            action_dict['next_states_prob'] = {}
            next_states = action_dict['next_states']
            rewards = action_dict['rewards']
            next_states_set = set(next_states)

            next_states_numpy = np.array(next_states, dtype=int)
            rewards_numpy = np.array(rewards)

            for next_state in next_states_set:
                next_state_idx = next_states_numpy == next_state
                next_state_prob = np.sum(next_state_idx) / len(next_states_numpy)
                avg_reward = np.sum(next_state_idx * rewards_numpy) / np.sum(next_state_idx)
                action_dict['next_states_prob'][next_state] = next_state_prob
                action_dict['avg_rewards'][next_state] = avg_reward

    return rb_dict


def is_correct_transition(state, action, next_state, real_env):
    real_env.env.env.state = real_env.env.env._obs_to_state(state)
    next_state_real, _, _, _ = real_env.env.step(action)
    return next_state_real == next_state


def plot_agent_behaviour(rb_dict, q_table, real_env):
    max_reward = -1e9
    min_reward = +1e9
    for state, state_dict in rb_dict.items():
        for action, action_dict in state_dict.items():
            min_reward = min(min_reward, min(action_dict['avg_rewards'].values()))
            max_reward = max(max_reward, max(action_dict['avg_rewards'].values()))

    print(min_reward)
    print(max_reward)

    for state, state_dict in rb_dict.items():
        x_state, y_state = idx_to_xy(state)

        for action, action_dict in state_dict.items():
            x_offset = 0
            y_offset = 0
            if action == G_UP:
                y_offset = +NS_OFFSET
            elif action == G_DOWN:
                y_offset = -NS_OFFSET
            elif action == G_RIGHT:
                x_offset = +NS_OFFSET
            elif action == G_LEFT:
                x_offset = -NS_OFFSET
            else:
                raise NotImplementedError('Unknown action: ' + str(action))

            next_states_prob = action_dict['next_states_prob']
            avg_rewards = action_dict['avg_rewards']

            n = len(next_states_prob)
            offset = [NS_DIFF*(elem-(n-1)/2) for elem in range(n)]

            for i, next_state in enumerate(sorted(next_states_prob)):
                x_pos = x_state + x_offset + (y_offset != 0) * offset[i]
                y_pos = y_state + y_offset + (y_offset == 0) * (-offset[i])

                mapped_reward = (avg_rewards[next_state]-min_reward)/(max_reward-min_reward)
                print('----')
                print(mapped_reward)
                print(next_states_prob[next_state])
                plot_filled_circle((x_pos,y_pos), NS_DIFF-0.02, intensity=next_states_prob[next_state])
                plot_filled_circle((x_pos, y_pos), NS_DIFF - 0.06, intensity=mapped_reward)
                if is_correct_transition(real_env=real_env, state=state, action=action, next_state=next_state):
                    text_color='r'
                else:
                    text_color='k'
                plt.gca().text(x_pos, y_pos, next_state, color=text_color, fontsize=FONTSIZE_SMALL, ha='center', va='center')
            # print(next_states_prob)
            # print(avg_rewards)



            #print('{} {}'.format(state, action))




if __name__ == "__main__":
    # for i in range(25):
    #     x,y = idx_to_xy(i)
    #     print('{} {} {} {}'.format(i, xy_to_idx((x,y)), x, y))
    #     #print(r.grid[x][y])

    dir = '/home/dingsda/master_thesis/learning_environments/agents/results/GTN_models'
    file_name = '2x3_7_9AFZ2G.pt'
    virtual_env, real_env, config, M, N, gtn_it = load_envs_and_config(dir=dir, file_name=file_name)


    replay_buffer_train_all = ReplayBuffer(state_dim=M*N, action_dim=4, device='cpu')
    replay_buffer_test_all = ReplayBuffer(state_dim=M*N, action_dim=4, device='cpu')
    q_tables = []
    for i in range(10):
        print(i)
        agent = select_agent(config=config, agent_name='QL')
        _, replay_buffer_train = agent.train(env=virtual_env, gtn_iteration=gtn_it)
        reward, replay_buffer_test = agent.test(env=real_env)
        replay_buffer_train_all.merge(replay_buffer_train)
        replay_buffer_test_all.merge(replay_buffer_test)
        print(reward)

    rb_dict_train = convert_replay_buffer(replay_buffer_train_all)
    #rb_dict_test = convert_replay_buffer(replay_buffer_test_all)

    fig, ax = plt.subplots(1,dpi=600)
    plot_tiles(length=0.9, n_tot=M*N)
    plot_tile_numbers(n_tot=M*N)
    plot_agent_behaviour(rb_dict=rb_dict_test, q_table=[], real_env=real_env)

    # plot_filled_rectangle((0,1), 0.2, 0.2)
    # plot_filled_rectangle((0,1), 0.1, 0)

    ax.axis('equal')
    #ax.axis('off')
    plt.show()