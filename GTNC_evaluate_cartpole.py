import datetime
import time
import sys
import traceback
import yaml
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import random
import numpy as np
import torch
from copy import deepcopy
from agents.GTN import GTN_Master
from automl.bohb_optim import run_bohb_parallel, run_bohb_serial
from communicate.tcp_master_selector import start_communication_thread


def seed_experiment(number):
    random.seed(number + int(time.time()))
    np.random.seed(number + int(time.time()))
    torch.manual_seed(number + int(time.time()))
    torch.cuda.manual_seed_all(number + int(time.time()))


class ExperimentWrapper():

    def get_bohb_parameters(self):
        params = {}
        params['min_budget'] = 1
        params['max_budget'] = 1
        params['eta'] = 2
        params['random_fraction'] = 1
        params['iterations'] = 10000

        return params

    def get_configspace(self):
        cs = CS.ConfigurationSpace()

        return cs

    def get_specific_config(self, cso, default_config, budget):
        config = deepcopy(default_config)
        return config

    def compute(self, working_dir, bohb_id, config_id, cso, budget, *args, **kwargs):
        with open("default_config_cartpole_syn_env.yaml", 'r') as stream:
            default_config = yaml.safe_load(stream)

        config = self.get_specific_config(cso, default_config, budget)

        print('----------------------------')
        print("START BOHB ITERATION")
        print('CONFIG: ' + str(config))
        print('CSO:    ' + str(cso))
        print('BUDGET: ' + str(budget))
        print('----------------------------')

        try:
            start_communication_thread(self.args)
            gtn = GTN_Master(config, bohb_id=bohb_id, bohb_working_dir=working_dir)
            _, score_list = gtn.run()
            score = len(score_list)
            error = ""
        except:
            score = float('Inf')
            score_list = []
            error = traceback.format_exc()
            print(error)

        info = {}
        info['error'] = str(error)
        info['score_list'] = str(score_list)

        print('----------------------------')
        print('FINAL SCORE: ' + str(score))
        print('SCORE LIST:  ' + str(score_list))
        print("END BOHB ITERATION")
        print('----------------------------')

        return {
            "loss": score,
            "info": info
        }


if __name__ == "__main__":
    x = datetime.datetime.now()
    run_id = 'GTNC_evaluate_cartpole_' + x.strftime("%Y-%m-%d-%H")

    if len(sys.argv) > 1:  # eg 5000 1
        print("if len(sys.argv) > 1:")

        for arg in sys.argv[1:]:
            print(arg)

        seed_experiment(number=int(sys.argv[1]))
        res = run_bohb_parallel(id=int(sys.argv[1]),
                                bohb_workers=1,  # this should be set to 1, otherwise the program does not execute
                                run_id=run_id,
                                experiment_wrapper=ExperimentWrapper())
    else:
        print("else CASE")

        seed_experiment(number=int(sys.argv[1]))
        res = run_bohb_serial(run_id=run_id,
                              experiment_wrapper=ExperimentWrapper())
