import torch
import os
import statistics
import numpy as np
import hpbandster.core.result as hpres
from envs.env_factory import EnvFactory
import matplotlib.pyplot as plt


# from gridworld.py
G_RIGHT = 0
G_LEFT = 1
G_DOWN = 2
G_UP = 3

LOG_DIR = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_cliff_2021-02-09-21_5'
MODEL_NUM = 50
SIMPLIFY = False

def idx_to_xy(idx, n):
    x = idx // n
    y = idx % n
    return y, -x


def xy_to_idx(xy, n):
    y, x = xy
    obs = -x * n + y
    return obs


def get_best_models_from_log(log_dir):
    if not os.path.isdir(log_dir):
        log_dir = log_dir.replace('nierhoff', 'dingsda')

    result = hpres.logged_results_to_HBS_result(log_dir)

    best_models = []

    for value in result.data.values():
        try:
            model_name = value.results[1.0]['info']['model_name']

            if not os.path.isfile(model_name):
                model_name = model_name.replace('nierhoff', 'dingsda')
            best_models.append(model_name)
        except:
            continue

    best_models.sort(key=lambda x: x[0])
    best_models = best_models[:MODEL_NUM]

    return best_models


def eval_models(log_dir):
    best_models = get_best_models_from_log(log_dir)

    info_dict = {}
    reward_dict = {}

    info_dict['mode'] = log_dir[-1]

    for model_file in best_models:
        reward_env, real_env, config = load_envs_and_config(model_file)
        info_dict['m'] = len(real_env.env.grid)
        info_dict['n'] = len(real_env.env.grid[0])

        for state in range(reward_env.get_state_dim()):
            for action in range(reward_env.get_action_dim()):
                reward_env.set_agent_params(same_action_num=1, gamma=config['agents']['ql']['gamma'])
                reward_env.env.real_env.env.state = reward_env.env.real_env.env._obs_to_state(state)
                reward_env.env.state = state
                next_state, reward, _ = reward_env.step(action=torch.tensor([action]))

                if SIMPLIFY:
                    if reward.item() > -50:
                        if (state, action) not in reward_dict:
                            reward_dict[(state, action)] = [reward.item()]
                        else:
                            reward_dict[(state, action)].append(reward.item())
                else:
                    if (state, action) not in reward_dict:
                        reward_dict[(state, action)] = [reward.item()]
                    else:
                        reward_dict[(state, action)].append(reward.item())


    return reward_dict, info_dict


def map_intensity_to_color(intensity):
    if intensity <= 0:
        return np.array([1, 0, 0])
    elif intensity <= 0.5:
        return np.array([1, 0, 0]) + 2 * intensity * np.array([0, 1, 0])
    elif intensity <= 1:
        return np.array([1, 1, 0]) + 2 * (intensity - 0.5) * np.array([-1, 0, 0])
    else:
        return np.array([0, 1, 0])


def draw_filled_polygon(state, action, n, intensity):
    x,y = idx_to_xy(state, n)

    if action == G_RIGHT:
        xs = [x+0.5, x+0.5, x]
        ys = [y-0.5, y+0.5, y]
    elif action == G_LEFT:
        xs = [x-0.5, x-0.5, x]
        ys = [y-0.5, y+0.5, y]
    elif action == G_DOWN:
        xs = [x-0.5, x+0.5, x]
        ys = [y-0.5, y-0.5, y]
    elif action == G_UP:
        xs = [x-0.5, x+0.5, x]
        ys = [y+0.5, y+0.5, y]

    plt.fill(xs, ys, facecolor=map_intensity_to_color(intensity))


def plot_models(reward_dict, info_dict):
    # create average reward_dict
    min_val = float('Inf')
    max_val = float('-Inf')
    reward_avg_dict = {}
    for key, value in reward_dict.items():
        reward_avg_dict[key] = statistics.mean(value)
        min_val = min(min_val, reward_avg_dict[key])
        max_val = max(max_val, reward_avg_dict[key])

    print(min_val)
    print(max_val)
    n = info_dict['n']
    mode = info_dict['mode']

    # plot individual rewards
    fig, ax = plt.subplots(dpi=600, figsize=(7, 2.5))

    for key, value in reward_avg_dict.items():
        intensity = (value-min_val) / (max_val-min_val)
        state = key[0]
        action = key[1]
        draw_filled_polygon(state, action, n, intensity)


    for i in range(5):
        plt.plot([-0.5, 11.5], [-i+0.5, -i+0.5], linewidth=0.5, color='black')
    for i in range(13):
        plt.plot([i-0.5, i-0.5], [0.5, -3.5], linewidth=0.5, color='black')

    # plot additional information
    x_water = [0.5, 10.5, 10.5, 0.5, 0.5]
    y_water = [-2.5, -2.5, -3.5, -3.5, -2.5]
    plt.plot(x_water, y_water, linewidth=2, color='black')
    plt.text(5.5, -3, 'cliff', size=12, color='black', ha='center', va='center')
    plt.text(0, -3, '(S)', size=12, ha='center', va='center')
    plt.text(11, -3, '(G)', size=12, ha='center', va='center')


    if SIMPLIFY:
        if mode == '1':
            plt.title('exclusive potential reward network (only rewards > -50)')
        elif mode =='2':
            plt.title('additive potential reward network (only rewards > -50)')
        elif mode =='5':
            plt.title('exclusive non-potential reward network (only rewards > -50)')
        elif mode =='6':
            plt.title('additive non-potential reward network (only rewards > -50)')
    else:
        if mode == '1':
            plt.title('exclusive potential reward network')
        elif mode == '2':
            plt.title('additive potential reward network')
        elif mode == '5':
            plt.title('exclusive non-potential reward network')
        elif mode == '6':
            plt.title('additive non-potential reward network')

    ax.axis('equal')
    ax.axis('off')

    if SIMPLIFY:
        plt.savefig('cliff_learned_rewards_' + str(mode) + '_simplified.svg', bbox_inches='tight')
    else:
        plt.savefig('cliff_learned_rewards_' + str(mode) + '.svg', bbox_inches='tight')

    plt.show()

def load_envs_and_config(model_file):
    save_dict = torch.load(model_file)

    config = save_dict['config']
    #config['device'] = 'cuda'
    config['envs']['Cliff']['solved_reward'] = 100000  # something big enough to prevent early out triggering

    env_factory = EnvFactory(config=config)
    reward_env = env_factory.generate_reward_env()
    reward_env.load_state_dict(save_dict['model'])
    real_env = env_factory.generate_real_env()

    return reward_env, real_env, config


if __name__ == "__main__":
    reward_dict, info_dict = eval_models(log_dir=LOG_DIR)
    plot_models(reward_dict=reward_dict, info_dict=info_dict)



