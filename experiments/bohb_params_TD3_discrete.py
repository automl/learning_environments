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

from agents.TD3_discrete_vary import TD3_discrete_vary
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

        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='init_episodes', lower=1, upper=20, log=True, default_value=10))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='batch_size', lower=64, upper=512, log=True, default_value=128))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='gamma', lower=0.001, upper=0.1, log=True, default_value=0.01))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='lr', lower=1e-6, upper=1e-1, log=True, default_value=5e-4))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='tau', lower=0.001, upper=0.1, log=True, default_value=0.01))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='policy_delay', lower=1, upper=5, log=False, default_value=2))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='hidden_size', lower=64, upper=512, log=True, default_value=128))
        cs.add_hyperparameter(
            CSH.CategoricalHyperparameter(name='activation_fn', choices=['relu', 'tanh', 'leakyrelu', 'prelu'], default_value='tanh'))

        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='action_std', lower=0.01, upper=10, log=True, default_value=0.1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='policy_std', lower=0.01, upper=10, log=True, default_value=0.2))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='policy_std_clip', lower=0.01, upper=10, log=True, default_value=0.5))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='gumbel_softmax_hard', choices=[True, False], default_value=False))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='gumbel_softmax_temp', lower=0.001, upper=10, log=True,
                                                             default_value=1.0))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='use_layer_norm', choices=[True, False], default_value=True))

        return cs

    def get_specific_config(self, cso, default_config, budget):
        config = deepcopy(default_config)

        config["agents"]["td3_discrete_vary"]["init_episodes"] = cso["init_episodes"]
        config["agents"]["td3_discrete_vary"]["batch_size"] = cso["batch_size"]
        config["agents"]["td3_discrete_vary"]["gamma"] = 1 - cso["gamma"]
        config["agents"]["td3_discrete_vary"]["lr"] = cso["lr"]
        config["agents"]["td3_discrete_vary"]["tau"] = cso["tau"]
        config["agents"]["td3_discrete_vary"]["policy_delay"] = cso["policy_delay"]
        config["agents"]["td3_discrete_vary"]["hidden_size"] = cso["hidden_size"]
        config["agents"]["td3_discrete_vary"]["activation_fn"] = cso["activation_fn"]
        config["agents"]["td3_discrete_vary"]["action_std"] = cso["action_std"]
        config["agents"]["td3_discrete_vary"]["policy_std"] = cso["policy_std"]
        config["agents"]["td3_discrete_vary"]["policy_std_clip"] = cso["policy_std_clip"]
        config["agents"]["td3_discrete_vary"]["gumbel_softmax_temp"] = cso["gumbel_softmax_temp"]
        config["agents"]["td3_discrete_vary"]["gumbel_softmax_hard"] = cso["gumbel_softmax_hard"]
        config["agents"]["td3_discrete_vary"]["use_layer_norm"] = cso["use_layer_norm"]

        return config

    def compute(self, working_dir, bohb_id, config_id, cso, budget, *args, **kwargs):
        with open("default_config_cartpole.yaml", 'r') as stream:
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

        # with BOHB, we want it to specify the variation of HPs
        config["agents"]["td3_discrete_vary"]["vary_hp"] = False

        td3 = TD3_discrete_vary(env=env,
                                min_action=env.get_min_action(),
                                max_action=env.get_max_action(),
                                config=config)


        score_list = []
        for _ in range(5):
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
    run_id = 'bohb_params_TD3_discrete_gumbel_temp_annealing_' + x.strftime("%Y-%m-%d-%H")

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
