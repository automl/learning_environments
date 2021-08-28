import datetime
import random
import sys
import time
import traceback
from copy import deepcopy

import ConfigSpace as CS
import numpy as np
import torch
import yaml

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
        with open("default_config_halfcheetah_td3_se_opt.yaml", 'r') as stream:
            default_config = yaml.safe_load(stream)

        config = self.get_specific_config(cso, default_config, budget)

        print('----------------------------')
        print("START BOHB ITERATION")
        print('CONFIG: ' + str(config))
        print('CSO:    ' + str(cso))
        print('BUDGET: ' + str(budget))
        print('----------------------------')

        try:
            gtn = GTN_Master(config, bohb_id=bohb_id, bohb_working_dir=working_dir)
            _, score_list, model_name = gtn.run()
            # score = -sorted(score_list)[-1]
            score = len(score_list)
            error = ""
        except:
            score = float('-Inf')
            score_list = []
            model_name = None
            error = traceback.format_exc()
            print(error)

        info = {}
        info['error'] = str(error)
        info['config'] = str(config)
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

    id = int(sys.argv[1])  # argument 1 := id , argument 2 := Number of BOHB workers? -> Number of Masters?? -> seems to be used as min BHOB workers in bhob_optim call
    bohb_workers = int(sys.argv[2])

    run_id = 'syn_env_learn_halfcheetah_' + x.strftime("%Y-%m-%d-%H")

    seed = id + int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    res = run_bohb_parallel(id=id,
                            bohb_workers=bohb_workers,
                            run_id=run_id,
                            experiment_wrapper=ExperimentWrapper())
