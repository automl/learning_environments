import random
import colorsys
import math

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
OUTLIER_PERC_WORST = 0.8
OUTLIER_PERC_BEST = 0.0


def analyze_bohb(log_dir):
    # load the example run from the log files
    result = hpres.logged_results_to_HBS_result(log_dir)

    # get all executed runs
    all_runs = result.get_all_runs()

    # get the 'dict' that translates config ids to the actual configurations
    id2conf = result.get_id2config_mapping()

    # Here is how you get he incumbent (best configuration)
    inc_id = result.get_incumbent_id()

    # let's grab the run on the highest budget
    inc_runs = result.get_runs_by_id(inc_id)
    inc_run = inc_runs[-1]

    # We have access to all information: the config, the loss observed during
    # optimization, and all the additional information
    inc_valid_score = inc_run.loss
    inc_config = id2conf[inc_id]['config']
    inc_info = inc_run['info']

    print('Best found configuration :' + str(inc_config))
    print('Score: ' + str(inc_valid_score))
    print('Info: ' + str(inc_info))
    # print('It achieved accuracies of %f (validation) and %f (test).' % (-inc_valid_score, inc_test_score))

    # # Let's plot the observed losses grouped by budget,
    # hpvis.losses_over_time(all_runs)
    #
    # # the number of concurent runs,
    # hpvis.concurrent_runs_over_time(all_runs)
    #
    # # and the number of finished runs.
    # hpvis.finished_runs_over_time(all_runs)
    #
    # # This one visualizes the spearman rank correlation coefficients of the losses
    # # between different budgets.
    # hpvis.correlation_across_budgets(result)
    #
    # # For model based optimizers, one might wonder how much the model actually helped.
    # # The next plot compares the performance of configs picked by the model vs. random ones
    # hpvis.performance_histogram_model_vs_random(all_runs, id2conf)

    result = remove_outliers(result)

    #result = filter_values(result)

    #print_configs_sorted_by_loss(result)

    #print_stats_per_value(result)

    #plot_accuracy_over_budget(result)

    plot_parallel_scatter(result)

    plt.show()


def print_configs_sorted_by_loss(result):
    lst = []

    for k1, v1 in result.data.items():
        for k2, v2 in v1.results.items():
            loss = v2['loss']
            config = v1.config
            lst.append((loss,config))

    lst.sort(key = lambda x: x[0])

    for elem in lst:
        print(elem)


def print_stats_per_value(result):

    # get all possible keys
    min_epoch = float('Inf')

    config_params = {}
    for value in result.data.values():
        for config_param, config_param_val in value.config.items():
            for epoch, epoch_result in value.results.items():
                try:
                    loss = epoch_result["loss"]

                    min_epoch = min(min_epoch, epoch)

                    if config_param in config_params.keys():
                        config_params[config_param].append((config_param_val, epoch, loss))
                    else:
                        config_params[config_param] = [(config_param_val, epoch, loss)]
                except:
                    print('Error in get_avg_per_value, continuing')

    for config_param, data in (dict(sorted(config_params.items()))).items():
        print(config_param)

        # get all unique possible values for each config parameter
        values = set(elem[0] for elem in data)
        values = sorted(list(values))

        if len(values) > 20:
            continue

        for value in values:
            losses = []
            for elem in data:
                val, epoch, loss = elem
                if val == value and epoch == min_epoch:
                    losses.append(loss)

            print('{}  {}  {} {}'.format(value, np.mean(losses), np.std(losses), len(losses)))


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
        worst_loss += 0.01*abs(worst_loss)
    else:
        worst_loss -= 0.01*abs(worst_loss)

    # remove NaN's
    for i in range(len(lut)):
        if not math.isfinite(lut[i][0]) or lut[i][0] == 0:
            lut[i][0] = worst_loss
            for key in result.data[lut[i][1]].results.keys():
                result.data[lut[i][1]].results[key]['loss'] = worst_loss
            #result.data.pop(elem[1], None)

    lut.sort(key = lambda x: x[0], reverse=REVERSE_LOSS)
    n_remove_worst = math.ceil(len(lut)*OUTLIER_PERC_WORST)
    n_remove_best = math.ceil(len(lut)*OUTLIER_PERC_BEST)

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
        config = value1.config

        rep_env_num = config['rep_env_num']
        ddqn_dropout = config['ddqn_dropout']
        # if not ddqn_dropout == 0:
        #     del_list.append(id)
        # if not rep_env_num == 5:
        #     del_list.append(id)

    for id in del_list:
        result.data.pop(id, None)

    return result


def plot_accuracy_over_budget(result):
    fig, ax = plt.subplots()

    # plot hyperband plot
    index = None
    color = None

    for key, value1 in result.data.items():
        if key[0] is not index:
            index = key[0]
            color = get_bright_random_color()

        try:
            x = []
            y = []
            for key2, value2 in value1.results.items():
                x.append(key2)
                y.append(value2["loss"])
            plt.semilogx(x, y, color=color)
        except:
            print('Error in plot_accuracy_over_budget, continuing')

    ax.set_title('Score for different configurations')
    ax.set_xlabel('epochs')
    ax.set_ylabel('score')


def plot_parallel_scatter(result):
    plt.subplots(dpi=300, figsize=(8, 4))

    ep_m = 1e9
    ep_M = -1e9
    loss_m = 1e9
    loss_M = -1e9

    # get all possible keys
    config_params = {}
    for value in result.data.values():
        for config_param, config_param_val in value.config.items():
            for epoch, epoch_result in value.results.items():
                try:
                    loss = epoch_result["loss"]
                    ep_m = min(ep_m, epoch)
                    ep_M = max(ep_M, epoch)
                    loss_m = min(loss_m, loss)
                    loss_M = max(loss_M, loss)

                    if config_param in config_params.keys():
                        config_params[config_param].append((config_param_val, epoch, loss))
                    else:
                        config_params[config_param] = [(config_param_val, epoch, loss)]
                except:
                    print('Error in plot_parallel_scatter, continuing')

    x_dev = 0.2
    r_min = 3
    r_max = 4
    alpha = 0.4
    text_x_offset = -0.1
    text_y_offset = -0.1
    size_text = 6

    index = 0
    for config_param, data in (dict(sorted(config_params.items()))).items():
        # get all unique possible values for each config parameter
        values = set(elem[0] for elem in data)
        values = sorted(list(values))

        n = len(data)
        xs = np.zeros(n)
        ys = np.zeros(n)
        rads = np.zeros(n)
        colors = np.zeros([n, 3])

        # extract common features
        for i in range(len(values)):
            for k in range(len(data)):
                if data[k][0] == values[i]:
                    ep = data[k][1]
                    acc = map_to_zero_one_range(data[k][2], loss_m, loss_M)
                    rads[k] = linear_interpolation(np.log(ep), np.log(ep_m), np.log(ep_M), r_min, r_max) ** 2
                    colors[k, :] = get_color(acc)

        # check for type (categorical,int,float,log)
        if type(values[0]) is bool:
            y_dev = x_dev / 2
            for i in range(len(values)):
                plt.text(index + text_x_offset, values[i] + text_y_offset, str(values[i]), rotation=90,
                         size=size_text)
                for k in range(len(data)):
                    if data[k][0] == values[i]:
                        xs[k] = index + np.random.uniform(-x_dev, x_dev)
                        ys[k] = values[i] + np.random.uniform(-y_dev, y_dev)

        elif type(values[0]) is str:
            y_dev = min(1 / len(values) / 2.5, x_dev / 2)
            for i in range(len(values)):
                plt.text(index + text_x_offset, i / (max(len(values) - 1, 1)) + text_y_offset, values[i],
                         rotation=90, size=size_text)
                for k in range(len(data)):
                    if data[k][0] == values[i]:
                        xs[k] = index + np.random.uniform(-x_dev, x_dev)
                        ys[k] = i / (max(len(values) - 1, 1)) + np.random.uniform(-y_dev, y_dev)

        elif type(values[0]) is int:
            y_dev = min(1 / len(values) / 2.5, x_dev / 2)

            plotAllStr = len(values) < 20

            if not plotAllStr:
                min_val = min(values)
                max_val = max(values)
                plt.text(index + text_x_offset, 0 + text_y_offset, str(f"{Decimal(min_val):.1E}"), rotation=90, size=size_text)
                plt.text(index + text_x_offset, 1 + text_y_offset, str(f"{Decimal(max_val):.1E}"), rotation=90, size=size_text)

            for i in range(len(values)):
                if plotAllStr:
                    plt.text(index + text_x_offset, i / (max(len(values) - 1, 1)), str(values[i]), rotation=90,
                             size=size_text)
                for k in range(len(data)):
                    if data[k][0] == values[i]:
                        xs[k] = index + np.random.uniform(-x_dev, x_dev)
                        ys[k] = i / (max(len(values) - 1, 1)) + np.random.uniform(-y_dev, y_dev)

        else:  # float
            min_val = min(values)
            max_val = max(values)

            # log scale if min/max value differs to much
            if max_val / min_val > 100:
                val050 = np.exp((np.log(min_val)+np.log(max_val))/2)
                for i in range(len(values)):
                    for k in range(len(data)):
                        if data[k][0] == values[i]:
                            xs[k] = index + np.random.uniform(-x_dev, x_dev)
                            ys[k] = linear_interpolation(np.log(data[k][0]), np.log(min_val), np.log(max_val), 0, 1)

            # linear scale
            else:
                val050 = linear_interpolation(0.50, 0, 1, min_val, max_val)
                for i in range(len(values)):
                    for k in range(len(data)):
                        if data[k][0] == values[i]:
                            xs[k] = index + np.random.uniform(-x_dev, x_dev)
                            ys[k] = linear_interpolation(np.log(data[k][0]), np.log(min_val), np.log(max_val), 0, 1)

            plt.text(index + text_x_offset, 0 + text_y_offset, str(f"{Decimal(min_val):.1E}"), rotation=90, size=size_text)
            plt.text(index + text_x_offset, 0.5 + text_y_offset, str(f"{Decimal(val050):.1E}"), rotation=90, size=size_text)
            plt.text(index + text_x_offset, 1 + text_y_offset, str(f"{Decimal(max_val):.1E}"), rotation=90, size=size_text)

        plt.scatter(xs, ys, s=rads, c=colors, alpha=alpha, edgecolors='none')
        index += 1

    plt.yticks([], [])
    plt.xticks(np.arange(index), (tuple(sorted(config_params.keys()))), rotation=90, fontsize=size_text)


def linear_interpolation(x, x0, x1, y0, y1):
    # linearly interpolate between two x/y values for a given x value
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

def map_to_zero_one_range(loss, loss_m, loss_M):
    if loss_M < 1 and loss_m > 0 and REVERSE_LOSS == False:
        # if we have already a loss in the [0,1] range, there is no need to normalize anything
        acc = loss
    elif loss_M < 0 and loss_m > -1 and REVERSE_LOSS == True:
        # if we have a loss in the [-1,0] range, simply revert its sign
        acc = -loss
    else:
        # normalize loss to the 0 (bad) - 1(good) range
        acc = (loss-loss_m) / (loss_M - loss_m)
        if REVERSE_LOSS:
            acc = 1-acc

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
    #log_dir = '../results/TD3_params_bohb_2020-07-07-12'
    #log_dir = '../results/GTN_params_reduced_bohb_2020-07-18-06-pen-latest-greatest2'
    log_dir = '../results/TD3_params_bohb_2020-07-07-12'
    analyze_bohb(log_dir)



