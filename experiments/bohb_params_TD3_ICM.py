import datetime
import random
import sys
import time
from copy import deepcopy

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
import torch
import yaml

from agents.TD3 import TD3
from automl.bohb_optim import run_bohb_parallel, run_bohb_serial
from envs.env_factory import EnvFactory


class ExperimentWrapper():
    def get_bohb_parameters(self):
        params = {}
        params['min_budget'] = 1
        params['max_budget'] = 2
        params['eta'] = 2
        params['iterations'] = 10000
        params['random_fraction'] = 0.3

        return params

    def get_configspace(self):
        cs = CS.ConfigurationSpace()

        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='lr', lower=1e-6, upper=1e-3, log=True, default_value=1e-4))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='beta', lower=0.001, upper=1.0, log=True, default_value=0.2))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='eta', lower=0.001, upper=1.0, log=True, default_value=0.5))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='feature_dim', lower=16, upper=256, log=True, default_value=64))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='hidden_size', lower=16, upper=256, log=True, default_value=128))

        return cs

    def get_specific_config(self, cso, default_config, budget):
        config = deepcopy(default_config)

        config["agents"]["icm"]["lr"] = cso["lr"]
        config["agents"]["icm"]["beta"] = cso["beta"]
        config["agents"]["icm"]["eta"] = cso["eta"]
        config["agents"]["icm"]["feature_dim"] = cso["feature_dim"]
        config["agents"]["icm"]["hidden_size"] = cso["hidden_size"]

        return config

    def compute(self, working_dir, bohb_id, config_id, cso, budget, *args, **kwargs):
        # with open("default_config_halfcheetah.yaml", 'r') as stream:
        #     default_config = yaml.safe_load(stream)
        with open("default_config_cmc.yaml", 'r') as stream:
            default_config = yaml.safe_load(stream)

        config = self.get_specific_config(cso, default_config, budget)
        print('----------------------------')
        print("START BOHB ITERATION")
        print('CONFIG: ' + str(config))
        print('CSO:    ' + str(cso))
        print('BUDGET: ' + str(budget))
        print('----------------------------')

        info = {}

        # generate environment
        env_fac = EnvFactory(config)
        env = env_fac.generate_real_env()

        td3 = TD3(env=env,
                  max_action=env.get_max_action(),
                  config=config,
                  icm=True)

        score_list = []
        for _ in range(10):
            rewards, _, _ = td3.train(env)
            score_i = len(rewards)
            score_list.append(score_i)

        score = np.mean(score_list)

        info['config'] = str(config)

        print('----------------------------')
        print('FINAL SCORE: ' + str(score))
        print("END BOHB ITERATION")
        print('----------------------------')

        return {
                "loss": score,
                "info": info
                }


if __name__ == "__main__":
    x = datetime.datetime.now()
    # run_id = 'bohb_params_td3_icm_hc_' + x.strftime("%Y-%m-%d-%H")
    run_id = 'bohb_params_td3_icm_cmc_' + x.strftime("%Y-%m-%d-%H")

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            print(arg)
        random.seed(int(sys.argv[1]) + int(time.time()))
        np.random.seed(int(sys.argv[1]) + int(time.time()))
        torch.manual_seed(int(sys.argv[1]) + int(time.time()))
        torch.cuda.manual_seed_all(int(sys.argv[1]) + int(time.time()))
        res = run_bohb_parallel(id=sys.argv[1],
                                bohb_workers=sys.argv[2],
                                run_id=run_id,
                                experiment_wrapper=ExperimentWrapper())
    else:
        random.seed(int(time.time()))
        np.random.seed(int(time.time()))
        torch.manual_seed(int(time.time()))
        torch.cuda.manual_seed_all(int(time.time()))
        res = run_bohb_serial(run_id=run_id,
                              experiment_wrapper=ExperimentWrapper())
