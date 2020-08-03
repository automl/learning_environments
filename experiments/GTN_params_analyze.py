import ast
import random
import colorsys
import math
import hpbandster.core.result as hpres
import matplotlib.pyplot as plt
from copy import deepcopy


# smallest value is best -> reverse_loss = True
# largest value is best -> reverse_loss = False
REVERSE_LOSS = True
OUTLIER_PERC = 0.5


def analyze_bohb(log_dir):
    # load the example run from the log files
    result = hpres.logged_results_to_HBS_result(log_dir)

    # get the 'dict' that translates config ids to the actual configurations
    id2conf = result.get_id2config_mapping()

    result = remove_outliers(result)
    result_w = deepcopy(result)
    result_wo = deepcopy(result)

    analyze_result_order(result, id2conf)

    get_w(result_w, spec="w")
    get_w(result_wo, spec="wo")

    print(len(result_w.data))
    print(len(result_wo.data))

    plot_accuracy_over_budget(result_w,  'with virtual env')
    plot_accuracy_over_budget(result_wo, 'without virtual env')

    plt.show()


def get_w(result, spec):
    del_list = []
    for key1, value1 in result.data.items():
        for key2, value2 in value1.results.items():
            order = ast.literal_eval(value2['info']['order'])
            if spec == "w":    # consider all runs with virtual env
                if 2 not in order:
                    del_list.append((key1, key2))
            elif spec == "wo":  # consider all runs with real env only
                if 2 in order:
                    del_list.append((key1, key2))

    for key in del_list:
        key1 = key[0]
        key2 = key[1]
        result.data[key1].results.pop(key2, None)
        if len(result.data[key1].results) == 0:
            result.data.pop(key1)


def analyze_result_order(result, id2conf):
    order_list = []
    w_2 = []
    wo_2 = []
    for key, value1 in result.data.items():
        for key2, value2 in value1.results.items():
            loss = value2['loss']
            order = ast.literal_eval(value2['info']['order'])
            timings = ast.literal_eval(value2['info']['timings'])
            different_envs = id2conf[key]['config']['gtn_different_envs']
            pretrain_agent = id2conf[key]['config']['gtn_pretrain_agent']
            lr = id2conf[key]['config']['em_lr']
            steps = id2conf[key]['config']['em_max_steps']
            step_size = id2conf[key]['config']['em_step_size']

            order_list.append((loss, order, timings, different_envs, pretrain_agent, lr, steps, step_size))

            # budget
            if key2 > 1:
                continue

            if 2 in order:
                w_2.append(loss)
            else:
                wo_2.append(loss)

    order_list.sort(key = lambda x: x[0])
    for elem in order_list:
        print(elem)

    print(sum(w_2)/len(w_2))
    print(sum(wo_2)/len(wo_2))


def plot_accuracy_over_budget(result, plot_str):
    fig, ax = plt.subplots(dpi=300)

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
            plt.semilogx(x, y, color=color, marker=".")
        except:
            print('Error in plot_accuracy_over_budget, continuing')

    ax.set_title(plot_str)
    ax.set_xlabel('epochs')
    ax.set_ylabel('score')


def get_bright_random_color():
    h, s, l = random.random(), 1, 0.5
    return colorsys.hls_to_rgb(h, l, s)


def remove_outliers(result):
    lut = []
    for key, value1 in result.data.items():
        for value2 in value1.results.values():
            if value2 == None:
                loss = float('nan')
            else:
                loss = value2['loss']
            lut.append((loss, key))

    # remove NaN's
    for elem in lut:
        if not math.isfinite(elem[0]):
            result.data.pop(elem[1], None)
    lut = [x for x in lut if math.isfinite(x[0])]

    lut.sort(key = lambda x: x[0], reverse=REVERSE_LOSS)
    n_remove = math.ceil(len(lut)*OUTLIER_PERC)

    # remove percentage of largest/smallest values
    for i in range(n_remove):
        elem = lut.pop(0)
        result.data.pop(elem[1], None)

    return result


if __name__ == '__main__':
    #log_dir = '../results/TD3_params_bohb_2020-07-07-12'
    #log_dir = '../results/GTN_params_reduced_bohb_2020-07-18-06-pen-latest-greatest2'
    log_dir = '../results'
    analyze_bohb(log_dir)



