import torch
import statistics
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

RESULT_FILE = '/home/dingsda/master_thesis/learning_environments/results/cliff_compare_reward_envs/heatmap_end_2.pt'
M = 4
N = 12

def idx_to_xy(idx, n):
    x = idx // n
    y = idx % n
    return y, -x


def xy_to_idx(xy, n):
    y, x = xy
    obs = -x * n + y
    return obs


def plot_models():
    save_dict = torch.load(RESULT_FILE)
    state_list = save_dict['state_list']
    mode = save_dict['mode']
    break_cond = save_dict['break']

    state_list_flat = [elem for states in state_list for elem in states]
    state_count = Counter(state_list_flat)

    # plot individual rewards
    fig, ax = plt.subplots(dpi=600, figsize=(7, 2.5))

    for idx, count in state_count.items():
        intensity = count / max(state_count.values())
        x, y = idx_to_xy(idx, N)

        xs = [x - 0.5, x - 0.5, x + 0.5, x + 0.5]
        ys = [y - 0.5, y + 0.5, y + 0.5, y - 0.5]

        color = np.array([1, 1, 1]) - intensity * np.array([0, 1, 1])
        plt.fill(xs, ys, facecolor=color)


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

    if mode == '2' and break_cond == 'solved':
        plt.title('additive potential reward network (solved)')
    elif mode == '2' and break_cond == 'end':
        plt.title('additive potential reward network (end of training)')
    elif mode == '6' and break_cond == 'solved':
        plt.title('additive non-potential reward network (solved)')
    elif mode == '6' and break_cond == 'end':
        plt.title('additive non-potential reward network (end of training)')

    ax.axis('equal')
    ax.axis('off')
    plt.savefig('cliff_heatmap_' + str(break_cond) + '_' + str(mode) + '.svg', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_models()



