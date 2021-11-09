import hpbandster.core.result as hpres
import matplotlib.pyplot as plt
import numpy as np
import ast
import statistics
from copy import deepcopy

LOG_DIRS = ['../results/2_thomas_results/GTNC_evaluate_cartpole_2020-12-04-12',
            '../results/2_thomas_results/GTNC_evaluate_acrobot_2020-11-28-16']
MAX_VALS = 40
STD_MULT = 1


def get_data():
    list_data = []
    for log_dir in LOG_DIRS:
        result = hpres.logged_results_to_HBS_result(log_dir)
        all_runs = result.get_all_runs()
        id2conf = result.get_id2config_mapping()

        # calculate avg. runtime
        ts = []
        for i, run in enumerate(all_runs):
            t_s = run['time_stamps']['started']
            t_f = run['time_stamps']['finished']
            ts.append(t_f - t_s)

            if i >= MAX_VALS:
                break

        print(log_dir)
        print('mean [s]: ' + str(statistics.mean(ts)))
        print('std [s]: ' + str(statistics.stdev(ts)))

        # copy data to list
        data = []

        for run in all_runs:
            avg_rewards = ast.literal_eval(run['info']['score_list'])
            # print(avg_rewards)

            config_id = run['config_id']

            # handle timeout cases (impute missing values)
            if avg_rewards[0] < -1e5 and avg_rewards[1] > -1e5:
                avg_rewards[0] = avg_rewards[1]
            for k in range(1, len(avg_rewards)):
                if avg_rewards[k] < -1e5 and avg_rewards[k - 1] > -1e5:
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

    return proc_data, list_data


def plot_data(proc_data, list_data, savefig_name):
    fig, ax = plt.subplots(dpi=600, figsize=(7, 5))
    colors = ['#1F77B4', '#FF7F0E']
    #
    # for mean, _ in data_w:
    #     plt.plot(mean_w)
    #     colors.append(plt.gca().lines[-1].get_color())

    thresh_data = [
        ([0, 200], [195, 195], '--', colors[0], 2),
        ([0, 200], [-100, -100], '--', colors[1], 2)
    ]

    for (mean, std), thresh in zip(proc_data, thresh_data):
        plt.plot(thresh[0], thresh[1], thresh[2], color=thresh[3], linewidth=thresh[4])
        plt.plot(mean, linewidth=2)

    # plt.plot([0, 200], [195, 195], '--', color=colors[0], linewidth=2)
    # plt.plot([0, 200], [-100, -100], '--', color=colors[1], linewidth=2)

    for mean, std in proc_data:
        plt.fill_between(x=range(len(mean)), y1=mean - std * STD_MULT, y2=mean + std * STD_MULT, alpha=0.2)

    for i, dat in enumerate(list_data):
        print(len(dat))
        for k, avg_rewards in enumerate(dat):
            if k >= 20:
                continue
            plt.plot(avg_rewards, linewidth=0.3, color=colors[i])

    plt.legend(['CartPole-v0', 'CartPole-v0 solved threshold', 'Acrobot-v1', 'Acrobot-v1 solved threshold'], loc='lower right')
    plt.xlim(0, 199)
    plt.xlabel('ES outer loop iteration')
    plt.ylabel('cumulative reward')
    plt.savefig(savefig_name, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    proc_data, list_data = get_data()
    plot_data(proc_data=proc_data, list_data=list_data, savefig_name='cartpole_acrobot_success.png')
