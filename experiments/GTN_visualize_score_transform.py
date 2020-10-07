import hpbandster.core.result as hpres
import matplotlib.pyplot as plt
import numpy as np
import ast
from copy import deepcopy

LOG_DIR = '../results/GTNC_evaluate_score_transform_2020-10-05-21'
MAX_VALS = 100
STD_MULT = 0.5
SOLVED_REWARD = 0.8

def get_data(with_mirrored_sampling, finish_after_solved):

    result = hpres.logged_results_to_HBS_result(LOG_DIR)
    all_runs = result.get_all_runs()
    id2conf = result.get_id2config_mapping()

    # copy data to list
    list_data = [[] for _ in range(7)]

    for run in all_runs:
        avg_rewards = ast.literal_eval(run['info']['score_list'])
        config_id = run['config_id']
        config = id2conf[config_id]['config']
        score_transform_type = config['gtn_score_transform_type']
        mirrored_sampling = config['gtn_mirrored_sampling']

        if mirrored_sampling == with_mirrored_sampling:
            list_data[score_transform_type].append(avg_rewards)

    # copy from list to numpy array
    proc_data = []

    n = len(list_data[0][0])
    for data in list_data:
        np_data = np.zeros([MAX_VALS,n])

        for i in range(len(np_data)):
            dat = data[i]

            if finish_after_solved:
                for k in range(1,len(dat)):
                    if dat[k-1] > SOLVED_REWARD:
                        dat[k] = dat[k-1]
            np_data[i] = np.array(dat)

        mean = np.mean(np_data,axis=0)
        std = np.std(np_data,axis=0)

        proc_data.append((mean,std))

    return proc_data


def plot_data(data_wo, data_w, savefig_name):
    fig, ax = plt.subplots(dpi=600, figsize=(5,4))
    colors = []

    for mean_w, _ in data_w:
        plt.plot(mean_w)
        colors.append(plt.gca().lines[-1].get_color())

    for i, data in enumerate(zip(data_wo, data_w)):
        mean_wo, std_wo = data[0]
        mean_w, std_w = data[1]
        plt.plot(mean_wo, linestyle=':', color=colors[i])
        plt.fill_between(x=range(len(mean_w)), y1=mean_w-std_w*STD_MULT, y2=mean_w+std_w*STD_MULT, color=colors[i], alpha=0.1)

    plt.legend(('linear transf.', 'rank transf.', 'NES', 'NES unnorm.', 'single best', 'all better', 'single better'), loc=2)
    plt.xlim(0,49)
    plt.ylim(-0.2, 1)
    plt.xlabel('ES iteration')
    plt.ylabel('average reward')
    plt.savefig(savefig_name)
    plt.show()

if __name__ == "__main__":
    # FIXME: with_mirrored_sampling flags are intentionally reversed to fix bug in GTN.py
    data_wo_mirr = get_data(with_mirrored_sampling=True, finish_after_solved=False)
    data_w_mirr = get_data(with_mirrored_sampling=False, finish_after_solved=False)
    plot_data(data_wo=data_wo_mirr, data_w=data_w_mirr, savefig_name='score_transform_no_finish.png')

    # FIXME: with_mirrored_sampling flags are intentionally reversed to fix bug in GTN.py
    data_wo_mirr = get_data(with_mirrored_sampling=True, finish_after_solved=True)
    data_w_mirr = get_data(with_mirrored_sampling=False, finish_after_solved=True)
    plot_data(data_wo=data_wo_mirr, data_w=data_w_mirr, savefig_name='score_transform_finish.png')
