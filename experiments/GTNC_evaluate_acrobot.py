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
        with open("default_config_acrobot_opt.yaml", 'r') as stream:
            default_config = yaml.safe_load(stream)

        config = self.get_specific_config(cso, default_config, budget)

        print('----------------------------')
        print("START BOHB ITERATION")
        print('CONFIG: ' + str(config))
        print('CSO:    ' + str(cso))
        print('BUDGET: ' + str(budget))
        print('----------------------------')

        try:
            gtn = GTN_Master(config, bohb_id=bohb_id)
            _, score_list, model_name = gtn.run()
            score = len(score_list)
            error = ""
        except:
            score = float('Inf')
            score_list = []
            model_name = None
            error = traceback.format_exc()
            print(error)

        info = {}
        info['error'] = str(error)
        info['score_list'] = str(score_list)
        info['model_name'] = str(model_name)

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
    run_id = 'GTNC_evaluate_acrobot_' + x.strftime("%Y-%m-%d-%H")

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            print(arg)
        random.seed(int(sys.argv[1])+int(time.time()))
        np.random.seed(int(sys.argv[1])+int(time.time()))
        torch.manual_seed(int(sys.argv[1])+int(time.time()))
        torch.cuda.manual_seed_all(int(sys.argv[1])+int(time.time()))
        res = run_bohb_parallel(id=int(sys.argv[1]),
                                bohb_workers=int(sys.argv[2]),
                                run_id=run_id,
                                experiment_wrapper=ExperimentWrapper())
    else:
        random.seed(int(time.time()))
        np.random.seed(int(time.time()))
        torch.manual_seed(int(time.time()))
        torch.cuda.manual_seed_all(int(time.time()))
        res = run_bohb_serial(run_id=run_id,
                              experiment_wrapper=ExperimentWrapper())
