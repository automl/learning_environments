import random
import colorsys
import math
import ast

import numpy as np
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis

from decimal import Decimal
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# smallest value is best -> reverse_loss = True
# largest value is best -> reverse_loss = False
REVERSE_LOSS = True
EXP_LOSS = 1
OUTLIER_PERC_WORST = 0.0
OUTLIER_PERC_BEST = 0.0
MIN_SUCCESS_REWARD = 0.8


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def analyze_bohb(log_dir):
    # load the example run from the log files
    result = hpres.logged_results_to_HBS_result(log_dir)

    result = transform_result(result, min_success_reward=MIN_SUCCESS_REWARD)
    # result = remove_outliers(result)

    plot_parallel_scatter(result, with_mirrored_sampling=False, with_nes_step_size=False)
    plot_parallel_scatter(result, with_mirrored_sampling=False, with_nes_step_size=True)
    plot_parallel_scatter(result, with_mirrored_sampling=True, with_nes_step_size=False)
    plot_parallel_scatter(result, with_mirrored_sampling=True, with_nes_step_size=True)


def transform_result(result, min_success_reward):
    for key, value1 in result.data.items():
        for value2 in value1.results.values():
            score_list = value2['info']['score_list']
            score_list = ast.literal_eval(score_list)
            for i in range(len(score_list)):
                if score_list[i] > min_success_reward:
                    value2['loss'] = i
                    break
    return result


def remove_outliers(result):
    lut = []
    for key, value1 in result.data.items():
        for value2 in value1.results.values():
            if value2 == None:
                loss = float('nan')
            else:
                loss = value2['loss']
            lut.append([loss, key])

    filtered_lut = [x for x in lut if math.isfinite(x[0])]
    worst_loss = sorted(filtered_lut, reverse=REVERSE_LOSS)[0][0]

    if REVERSE_LOSS:
        worst_loss += 0.01 * abs(worst_loss)
    else:
        worst_loss -= 0.01 * abs(worst_loss)

    # remove NaN's
    for i in range(len(lut)):
        if not math.isfinite(lut[i][0]) or lut[i][0] == 0:
            lut[i][0] = worst_loss
            for key in result.data[lut[i][1]].results.keys():
                result.data[lut[i][1]].results[key]['loss'] = worst_loss
            # result.data.pop(elem[1], None)

    lut.sort(key=lambda x: x[0], reverse=REVERSE_LOSS)
    n_remove_worst = math.ceil(len(lut) * OUTLIER_PERC_WORST)
    n_remove_best = math.ceil(len(lut) * OUTLIER_PERC_BEST)

    # remove percentage of worst values
    for i in range(n_remove_worst):
        elem = lut.pop(0)
        result.data.pop(elem[1], None)

    # remove percentage of best values
    for i in range(n_remove_best):
        elem = lut.pop()
        result.data.pop(elem[1], None)

    return result


def filter_values(result):
    del_list = []
    for key, value1 in result.data.items():
        id = key
        if value1.config['gtn_score_transform_type'] != 7:
            del_list.append(id)
        # print('++++')
        # print(key)
        # print('----')
        # print(value1)

    for id in del_list:
        result.data.pop(id, None)

    return result


def plot_parallel_scatter(result, with_mirrored_sampling, with_nes_step_size):
    plt.subplots(dpi=300, figsize=(4, 4))

    min_step_size = 1e9
    max_step_size = -1e9

    # get all possible keys
    values = [[] for _ in range(8)]
    for value in result.data.values():
        config = value.config
        mirrored_sampling = str2bool(config['gtn_mirrored_sampling'])
        nes_step_size = str2bool(config['gtn_nes_step_size'])
        score_transform_type = config['gtn_score_transform_type']
        step_size = config['gtn_step_size']

        for value2 in value.results.values():
            loss = value2['loss']

            if mirrored_sampling == with_mirrored_sampling and nes_step_size == with_nes_step_size:
                values[score_transform_type].append((step_size, loss))

                min_step_size = min(min_step_size, step_size)
                max_step_size = max(max_step_size, step_size)

    loss_m = 0
    loss_M = 50

    x_dev = 0.2
    rad = 25
    alpha = 0.4
    text_x_offset = 0
    text_y_offset = 0
    size_text = 6

    xs = []
    ys = []

    for i in range(len(values)):
        xs = np.zeros(len(values[i]))
        ys = np.zeros(len(values[i]))
        colors = np.zeros([len(values[i]), 3])

        # log scale if min/max value differs to much
        if max_step_size / min_step_size > 10:
            for k in range(len(values[i])):
                step_size, loss = values[i][k]
                xs[k] = i + 1 + np.random.uniform(-x_dev, x_dev)
                ys[k] = linear_interpolation(np.log(step_size), np.log(min_step_size), np.log(max_step_size), 0, 1)
        # linear scale
        else:
            for k in range(len(values[i])):
                step_size, loss = values[i][k]
                xs[k] = i + 1 + np.random.uniform(-x_dev, x_dev)
                ys[k] = linear_interpolation(step_size, min_step_size, max_step_size, 0, 1)

        for k in range(len(values[i])):
            step_size, loss = values[i][k]
            acc = map_to_zero_one_range(loss, loss_m, loss_M)
            colors[k, :] = get_color(acc)

        plt.scatter(xs, ys, s=rad, c=colors, alpha=alpha, edgecolors='none')

    if max_step_size / min_step_size > 10:
        val050 = np.exp((np.log(min_step_size) + np.log(max_step_size)) / 2)
    else:
        val050 = linear_interpolation(0.50, 0, 1, min_step_size, max_step_size)

    if with_nes_step_size:
        nes_string = 'w/ NES step size'
    else:
        nes_string = 'w/o NES step size'

    if with_mirrored_sampling:
        mir_string = 'w/ mirrored sampling'
    else:
        mir_string = 'w/o mirrored sampling'

    plt.title(mir_string + ', ' + nes_string)
    plt.yticks([0, 0.5, 1], [str(f"{Decimal(min_step_size):.1E}"), str(f"{Decimal(val050):.1E}"), str(f"{Decimal(max_step_size):.1E}")])
    plt.xticks(np.arange(8) + 1, ('linear transf.', 'rank transf.', 'NES', 'NES unnorm.', 'single best', 'single better', 'all better 1', 'all better 2'), rotation=90)

    plt.show()


def linear_interpolation(x, x0, x1, y0, y1):
    # linearly interpolate between two x/y values for a given x value
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0 + 1e-9)


def map_to_zero_one_range(loss, loss_m, loss_M):
    if loss_M < 1 and loss_m > 0 and REVERSE_LOSS == False:
        # if we have already a loss in the [0,1] range, there is no need to normalize anything
        acc = loss
    elif loss_M < 0 and loss_m > -1 and REVERSE_LOSS == True:
        # if we have a loss in the [-1,0] range, simply revert its sign
        acc = -loss
    else:
        # normalize loss to the 0 (bad) - 1(good) range
        acc = (loss - loss_m) / (loss_M - loss_m)
        if REVERSE_LOSS:
            acc = 1 - acc

    acc = acc ** EXP_LOSS

    return acc


def get_color(acc):
    if acc <= 0:
        return np.array([[1, 0, 0]])
    elif acc <= 0.5:
        return np.array([[1, 0, 0]]) + 2 * acc * np.array([[0, 1, 0]])
    elif acc <= 1:
        return np.array([[1, 1, 0]]) + 2 * (acc - 0.5) * np.array([[-1, 0, 0]])
    else:
        return np.array([[0, 1, 0]])


def get_bright_random_color():
    h, s, l = random.random(), 1, 0.5
    return colorsys.hls_to_rgb(h, l, s)


if __name__ == '__main__':
    # log_dir = '../results/TD3_params_bohb_2020-07-07-12'
    # log_dir = '../results/GTN_params_reduced_bohb_2020-07-18-06-pen-latest-greatest2'
    log_dir = '../results/GTNC_params_bohb_gridworld_initial_2020-11-10-12'
    analyze_bohb(log_dir)
