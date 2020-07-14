import ast
import math

import numpy as np
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis

from decimal import Decimal
import matplotlib.pyplot as plt


# smallest value is best -> reverse_loss = True
# largest value is best -> reverse_loss = False
REVERSE_LOSS = True
EXP_LOSS = 2
OUTLIER_PERC = 0.2


def analyze_bohb(log_dir):
    # load the example run from the log files
    result = hpres.logged_results_to_HBS_result(log_dir)

    result = remove_outliers(result)

    analyze_result_order(result)


def analyze_result_order(result):
    order_list = []
    w_3 = []
    wo_3 = []
    for key, value1 in result.data.items():
        for value2 in value1.results.values():
            loss = value2['loss']
            order = ast.literal_eval(value2['info']['order'])
            order_list.append((loss, order))

            if 3 in order:
                w_3.append(loss)
            else:
                wo_3.append(loss)

    order_list.sort(key = lambda x: x[0])
    for elem in order_list:
        print(elem)

    print(sum(w_3)/len(w_3))
    print(sum(wo_3)/len(wo_3))


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
    log_dir = '../results/GTN_params_reduced_bohb_2020-07-13-21-with-order'
    analyze_bohb(log_dir)



