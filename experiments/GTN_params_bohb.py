import datetime
import sys
import traceback
import yaml
import random
import numpy as np
import torch
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from copy import deepcopy
from agents.GTN import GTN
from automl.bohb_optim import run_bohb_parallel, run_bohb_serial


class ExperimentWrapper():
    def get_bohb_parameters(self):
        params = {}
        params['seed'] = 42
        params['min_budget'] = 1
        params['max_budget'] = 4
        params['eta'] = 2
        params['iterations'] = 1000

        return params


    def get_configspace(self):
        cs = CS.ConfigurationSpace()

        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='gtn_max_iterations', lower=3, upper=10, log=False, default_value=5))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='gtn_step_size', lower=0.1, upper=1, log=True, default_value=0.2))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='gtn_mod_zero_rate', lower=2, upper=5, log=False, default_value=3))

        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='td3_mod_delay', choices=[1, 2, 4], default_value=1))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='td3_mod_type', lower=0, upper=5, log=False, default_value=0))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='td3_mod_grad_type', lower=1, upper=2, log=False, default_value=1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_mod_grad_step_size', lower=1e-3, upper=0.5, log=True, default_value=1e-2))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='td3_mod_grad_steps', lower=1, upper=3, log=False, default_value=1))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='td3_mod_noise_type', lower=1, upper=2, log=False, default_value=1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_mod_noise_std', lower=0.01, upper=1, log=True, default_value=0.1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_mod_add_factor', lower=0.01, upper=0.5, log=True, default_value=0.1))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='td3_mod_mult_const', choices=[False, True], default_value=False))

        return cs


    def get_specific_config(self, cso, default_config, budget):
        config = deepcopy(default_config)

        env_name = config["env_name"]
        config["render_env"] = False

        config["agents"]['gtn']['max_iterations'] = cso["gtn_max_iterations"]
        config["agents"]['gtn']['step_size'] = cso["gtn_step_size"]
        config["agents"]['gtn']['mod_zero_rate'] = cso["gtn_mod_zero_rate"]

        config["agents"]["td3"]["mod_delay"] = cso["td3_mod_delay"]
        config["agents"]["td3"]["mod_type"] = cso["td3_mod_type"]
        config["agents"]["td3"]["mod_grad_type"] = cso["td3_mod_grad_type"]
        config["agents"]["td3"]["mod_grad_step_size"] = cso["td3_mod_grad_step_size"]
        config["agents"]["td3"]["mod_grad_steps"] = cso["td3_mod_grad_steps"]
        config["agents"]["td3"]["mod_noise_type"] = cso["td3_mod_noise_type"]
        config["agents"]["td3"]["mod_noise_std"] = cso["td3_mod_noise_std"]
        config["agents"]["td3"]["mod_add_factor"] = cso["td3_mod_add_factor"]
        config["agents"]["td3"]["mod_mult_const"] = cso["td3_mod_mult_const"]

        return config


    def compute(self, working_dir, config_id, cso, budget, *args, **kwargs):
        with open("default_config.yaml", 'r') as stream:
            default_config = yaml.safe_load(stream)

        config = self.get_specific_config(cso, default_config, budget)

        print('----------------------------')
        print("START BOHB ITERATION")
        print('CONFIG: ' + str(config))
        print('CSO:    ' + str(cso))
        print('BUDGET: ' + str(budget))
        print('----------------------------')

        info = {}
        order = None
        timings = None
        episodes_till_solved = None
        error = ""
        try:
            gtn = GTN(config)
            order, timings = gtn.update()
            score, episodes_till_solved = gtn.test()
        except:
            score = float('Inf')
            error = traceback.format_exc()
            print(error)

        info['config'] = str(config)
        info['order'] = str(order)
        info['timings'] = str(timings)
        info['episodes_till_solved'] = str(episodes_till_solved)
        info['error'] = str(error)

        print('----------------------------')
        print('FINAL SCORE: ' + str(score))
        print('TIMINGS:     ' + str(timings))
        print("END BOHB ITERATION")
        print('----------------------------')

        return {
            "loss": score,
            "info": info
        }


if __name__ == "__main__":
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    x = datetime.datetime.now()
    run_id = 'GTN_params_bohb_' + x.strftime("%Y-%m-%d-%H")

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            print(arg)
        res = run_bohb_parallel(id=sys.argv[1],
                                bohb_workers=sys.argv[2],
                                run_id=run_id,
                                experiment_wrapper=ExperimentWrapper())
    else:
        res = run_bohb_serial(run_id=run_id,
                              experiment_wrapper=ExperimentWrapper())
