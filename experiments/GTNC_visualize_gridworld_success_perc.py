import hpbandster.core.result as hpres
import matplotlib.pyplot as plt
import numpy as np
import ast
from copy import deepcopy

LOG_DIRS = ['../results/GTNC_evaluate_gridworld_2x2_2020-10-05-19',
            '../results/GTNC_evaluate_gridworld_2x3_2020-10-05-19',
            '../results/GTNC_evaluate_gridworld_3x3_2020-10-05-19']
MAX_VALS = 100
STD_MULT = 0.5
SOLVED_REWARD = 0.8


def get_data(finish_after_solved):
    list_data = []
    for log_dir in LOG_DIRS:
        result = hpres.logged_results_to_HBS_result(log_dir)
        all_runs = result.get_all_runs()
        id2conf = result.get_id2config_mapping()

        # copy data to list
        data = []

        for run in all_runs:
            avg_rewards = ast.literal_eval(run['info']['score_list'])
            config_id = run['config_id']
            config = id2conf[config_id]['config']

            if finish_after_solved:
                for k in range(1, len(avg_rewards)):
                    if avg_rewards[k - 1] > SOLVED_REWARD:
                        avg_rewards[k] = avg_rewards[k - 1]

            data.append(avg_rewards)
        list_data.append(data)

    # copy from list to numpy array
    proc_data = []

    n = len(list_data[0][0])
    for data in list_data:
        np_data = np.zeros([MAX_VALS, n])

        for i in range(len(np_data)):
            np_data[i] = np.array(data[i])

        mean = np.mean(np_data, axis=0)
        std = np.std(np_data, axis=0)

        proc_data.append((mean, std))

    return proc_data


def plot_data(data, savefig_name):
    fig, ax = plt.subplots(dpi=600, figsize=(5, 4))
    colors = []
    #
    # for mean, _ in data_w:
    #     plt.plot(mean_w)
    #     colors.append(plt.gca().lines[-1].get_color())

    for i, data in enumerate(data):
        mean, std = data
        plt.plot(mean)
        plt.fill_between(x=range(len(mean)), y1=mean - std * STD_MULT, y2=mean + std * STD_MULT, alpha=0.1)

    plt.legend(('2x2 grid world', '2x3 grid world', '3x3 grid world'))
    plt.xlim(0, 49)
    plt.ylim(-0.2, 1)
    plt.xlabel('ES iteration')
    plt.ylabel('average reward')
    plt.savefig(savefig_name)
    plt.show()


if __name__ == "__main__":
    data = get_data(finish_after_solved=False)
    plot_data(data, savefig_name='gridworld_success_no_finish.png')

    data = get_data(finish_after_solved=True)
    plot_data(data, savefig_name='gridworld_success_finish.png')
