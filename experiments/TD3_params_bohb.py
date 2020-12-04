import datetime
import sys
import yaml
import random
import time
import numpy as np
import torch
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from copy import deepcopy
from agents.TD3 import TD3
from envs.env_factory import EnvFactory
from automl.bohb_optim import run_bohb_parallel, run_bohb_serial


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
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='batch_size', lower=64, upper=512, log=True, default_value=256))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='gamma', lower=0.001, upper=0.1, log=True, default_value=0.01))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='lr', lower=1e-4, upper=1e-1, log=True, default_value=3e-4))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='weight_decay', lower=1e-12, upper=1e-4, log=True, default_value=1e-10))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='tau', lower=0.001, upper=0.1, log=True, default_value=0.01))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='policy_delay', lower=1, upper=5, log=False, default_value=2))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='rb_size', lower=1000, upper=1000000, log=True, default_value=100000))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='hidden_size', lower=64, upper=512, log=True, default_value=224))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='activation_fn', choices=['relu', 'tanh', 'leakyrelu', 'prelu'], default_value='relu'))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='weight_norm', choices=[False, True], default_value=True))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='action_std', lower=0.01, upper=10, log=True, default_value=0.1))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='early_out_num', lower=1, upper=5, log=True, default_value=3))

        return cs


    def get_specific_config(self, cso, default_config, budget):
        config = deepcopy(default_config)

        config["agents"]["td3"]["init_episodes"] = cso["init_episodes"]
        config["agents"]["td3"]["batch_size"] = cso["batch_size"]
        config["agents"]["td3"]["gamma"] = 1-cso["gamma"]
        config["agents"]["td3"]["lr"] = cso["lr"]
        config["agents"]["td3"]["weight_decay"] = cso["weight_decay"]
        config["agents"]["td3"]["tau"] = cso["tau"]
        config["agents"]["td3"]["policy_delay"] = cso["policy_delay"]
        config["agents"]["td3"]["rb_size"] = cso["rb_size"]
        config["agents"]["td3"]["hidden_size"] = cso["hidden_size"]
        config["agents"]["td3"]["activation_fn"] = cso["activation_fn"]
        config["agents"]["td3"]["weight_norm"] = cso["weight_norm"]
        config["agents"]["td3"]["action_std"] = cso["action_std"]
        config["agents"]["td3"]["early_out_num"] = cso["early_out_num"]

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

        info = {}

        # generate environment
        env_fac = EnvFactory(config)
        env = env_fac.generate_real_env()

        td3 = TD3(env=env,
                  max_action=env.get_max_action(),
                  config=config)
        rewards, _ = td3.train(env)
        score = len(rewards)

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
    run_id = 'TD3_cartpole_params_bohb_' + x.strftime("%Y-%m-%d-%H")

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            print(arg)
        random.seed(int(sys.argv[1])+int(time.time()))
        np.random.seed(int(sys.argv[1])+int(time.time()))
        torch.manual_seed(int(sys.argv[1])+int(time.time()))
        torch.cuda.manual_seed_all(int(sys.argv[1])+int(time.time()))
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
