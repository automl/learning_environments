import os
import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from envs.env_factory import EnvFactory
from agents.agent_utils import select_agent
from utils import ReplayBuffer, print_abs_param_sum


# from gridworld.py
G_RIGHT = 0
G_LEFT = 1
G_DOWN = 2
G_UP = 3

NEXT_STATE_PROB_FACTOR = 2
INTENSITY_FACTOR = 0.8

FONTSIZE_LARGE = 14
FONTSIZE_SMALL = 12

NS_OFFSET = 0.32
NS_DIFF = 0.22

RADIUS_NEXT_STATE_PERC = 0.2
RADIUS_REWARD = 0.15
RADIUS_Q = 0.1

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
    rect = plt.Circle(center, diameter/2, edgecolor='k', linewidth=0.2, facecolor=map_intensity_to_color(intensity), fill=True)
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
                next_state_prob = np.sum(next_state_idx) / len(states) * NEXT_STATE_PROB_FACTOR
                avg_reward = np.sum(next_state_idx * rewards_numpy) / np.sum(next_state_idx)
                action_dict['next_states_prob'][next_state] = next_state_prob
                action_dict['avg_rewards'][next_state] = avg_reward

    return rb_dict


def is_correct_transition(state, action, next_state, real_env):
    real_env.env.env.state = real_env.env.env._obs_to_state(state)
    next_state_real, _, _, _ = real_env.env.step(action)
    return next_state_real == next_state


def plot_agent_behaviour(rb_dict, q_table, real_env):
    min_reward = +1e9
    max_reward = -1e9
    for state, state_dict in rb_dict.items():
        for action, action_dict in state_dict.items():
            min_reward = min(min_reward, min(action_dict['avg_rewards'].values()))
            max_reward = max(max_reward, max(action_dict['avg_rewards'].values()))

    # print(min_reward)
    # print(max_reward)

    min_q = +1e9
    max_q = -1e9

    for q_vals in q_table:
        min_q = min(min_q, min(q_vals))
        max_q = max(max_q, max(q_vals))

    # print(min_q)
    # print(max_q)

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
            #print('{} {} {}'.format(state, action, next_states_prob))

            for i, next_state in enumerate(sorted(next_states_prob)):
                x_pos = x_state + x_offset + (y_offset != 0) * offset[i]
                y_pos = y_state + y_offset + (y_offset == 0) * (-offset[i])

                mapped_reward = (avg_rewards[next_state]-min_reward)/(max_reward-min_reward)
                mapped_q = (q_table[state][action] - min_q) / (max_q - min_q)
                # print('----')
                # print(mapped_reward)
                # print(next_states_prob[next_state])
                plot_filled_circle((x_pos,y_pos), RADIUS_NEXT_STATE_PERC, intensity=next_states_prob[next_state]*INTENSITY_FACTOR)
                plot_filled_circle((x_pos, y_pos), RADIUS_REWARD, intensity=mapped_reward*INTENSITY_FACTOR)
                plot_filled_circle((x_pos, y_pos), RADIUS_Q, intensity=mapped_q*INTENSITY_FACTOR)
                if is_correct_transition(real_env=real_env, state=state, action=action, next_state=next_state):
                    text_color='r'
                else:
                    text_color='k'
                plt.gca().text(x_pos, y_pos-0.008, next_state, color=text_color, fontsize=FONTSIZE_SMALL, ha='center', va='center')
            # print(next_states_prob)
            # print(avg_rewards)


def merge_q_tables(q_tables):
    n = len(q_tables)
    q_table = [[0]*len(q_tables[0][0]) for _ in range(len(q_tables[0]))]
    for table in q_tables:
        print(table)
        for i in range(len(table)):
            for k in range(len(table[0])):
                q_table[i][k] += table[i][k]
    for i in range(len(q_table)):
        for k in range(len(q_table[0])):
            q_table[i][k] /= n

    return q_table



if __name__ == "__main__":
    # for i in range(25):
    #     x,y = idx_to_xy(i)
    #     print('{} {} {} {}'.format(i, xy_to_idx((x,y)), x, y))
    #     #print(r.grid[x][y])

    dir = '/home/dingsda/master_thesis/learning_environments/results/GTN_models_evaluate_gridworld'
    file_name = '2x3_25_DZCBB9.pt'
    virtual_env, real_env, config, M, N, gtn_it = load_envs_and_config(dir=dir, file_name=file_name)

    replay_buffer_train_all = ReplayBuffer(state_dim=1, action_dim=1, device='cpu')
    replay_buffer_test_all = ReplayBuffer(state_dim=1, action_dim=1, device='cpu')
    q_tables = []
    for i in range(10):
        print(i)
        agent = select_agent(config=config, agent_name='QL')
        _, replay_buffer_train = agent.train(env=virtual_env, gtn_iteration=gtn_it)
        reward, replay_buffer_test = agent.test(env=real_env)
        replay_buffer_train_all.merge_buffer(replay_buffer_train)
        replay_buffer_test_all.merge_buffer(replay_buffer_test)
        q_tables.append(agent.q_table)

    q_table = merge_q_tables(q_tables)

    rb_dict_train = convert_replay_buffer(replay_buffer_train_all)
    rb_dict_test = convert_replay_buffer(replay_buffer_test_all)

    fig, ax = plt.subplots(dpi=600)
    plot_tiles(length=0.9, n_tot=M*N)
    plot_tile_numbers(n_tot=M*N)
    plot_agent_behaviour(rb_dict=rb_dict_train, q_table=q_table, real_env=real_env)
    # plot_filled_rectangle((0,1), 0.2, 0.2)
    # plot_filled_rectangle((0,1), 0.1, 0)
    ax.axis('equal')
    ax.axis('off')
    plt.savefig('gridworld_train.eps')

    fig, ax = plt.subplots(dpi=600)
    plot_tiles(length=0.9, n_tot=M*N)
    plot_tile_numbers(n_tot=M*N)
    plot_agent_behaviour(rb_dict=rb_dict_test, q_table=q_table, real_env=real_env)
    # plot_filled_rectangle((0,1), 0.2, 0.2)
    # plot_filled_rectangle((0,1), 0.1, 0)
    ax.axis('equal')
    ax.axis('off')
    plt.savefig('gridworld_test.eps')

    plt.show()