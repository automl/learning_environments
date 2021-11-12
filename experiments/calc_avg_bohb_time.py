import statistics
import hpbandster.core.result as hpres

# smallest value is best -> reverse_loss = True
# largest value is best -> reverse_loss = False
REVERSE_LOSS = True
EXP_LOSS = 1
OUTLIER_PERC_WORST = 0.1
OUTLIER_PERC_BEST = 0.0


def analyze_bohb(log_dir):
    # load the example run from the log files
    result = hpres.logged_results_to_HBS_result(log_dir)

    # get all executed runs
    all_runs = result.get_all_runs()


if __name__ == '__main__':
    # load the example run from the log files
    result = hpres.logged_results_to_HBS_result('../results/GTNC_evaluate_cmc_subopt_2021-01-21-09_5')

    # get all executed runs
    all_runs = result.get_all_runs()

    t_arr = []
    for dat in result.data.values():
        for time_stamp in dat.time_stamps.values():
            ts = time_stamp['started']
            te = time_stamp['finished']
            if te - ts > 60:
                t_arr.append(te - ts)

    print(statistics.mean(t_arr))
