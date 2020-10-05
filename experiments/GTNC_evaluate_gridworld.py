import datetime
import sys
import traceback
import yaml
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import random
import numpy as np
import torch
from copy import deepcopy
from agents.GTN import GTN_Master, GTN_Worker
from automl.bohb_optim import run_bohb_parallel, run_bohb_serial


class ExperimentWrapper():
    def get_bohb_parameters(self):
        params = {}
        params['seed'] = 42
        params['min_budget'] = 1
        params['max_budget'] = 1
        params['eta'] = 2
        params['random_fraction'] = 1
        params['iterations'] = 15

        return params


    def get_configspace(self):
        cs = CS.ConfigurationSpace()

        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='0', lower=1e-2, upper=1, log=True, default_value=1e-1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='1', lower=1e-2, upper=1, log=True, default_value=1e-1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='2', lower=1e-2, upper=1, log=True, default_value=1e-1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='3', lower=1e-2, upper=1, log=True, default_value=1e-1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='4', lower=1e-2, upper=1, log=True, default_value=1e-1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='5', lower=1e-2, upper=1, log=True, default_value=1e-1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='6', lower=1e-2, upper=1, log=True, default_value=1e-1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='7', lower=1e-2, upper=1, log=True, default_value=1e-1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='8', lower=1e-2, upper=1, log=True, default_value=1e-1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='9', lower=1e-2, upper=1, log=True, default_value=1e-1))

        return cs


    def get_specific_config(self, cso, default_config, budget):
        config = deepcopy(default_config)
        return config


    def compute(self, working_dir, bohb_id, config_id, cso, budget, *args, **kwargs):
        with open("default_config.yaml", 'r') as stream:
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
    run_id = 'GTNC_evaluate_gridworld_2x3_' + x.strftime("%Y-%m-%d-%H")

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            print(arg)
        random.seed(int(sys.argv[1]))
        np.random.seed(int(sys.argv[1]))
        torch.manual_seed(int(sys.argv[1]))
        torch.cuda.manual_seed_all(int(sys.argv[1]))
        res = run_bohb_parallel(id=int(sys.argv[1]),
                                bohb_workers=int(sys.argv[2]),
                                run_id=run_id,
                                experiment_wrapper=ExperimentWrapper())
    else:
        res = run_bohb_serial(run_id=run_id,
                              experiment_wrapper=ExperimentWrapper())
