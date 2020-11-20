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
        params['max_budget'] = 3
        params['eta'] = 3
        params['random_fraction'] = 0.3
        params['iterations'] = 10000

        return params


    def get_configspace(self):
        cs = CS.ConfigurationSpace()

        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='gtn_score_transform_type', lower=0, upper=7, log=False, default_value=7))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='gtn_step_size', lower=0.1, upper=1, log=True, default_value=0.5))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='gtn_mirrored_sampling', choices=[False, True], default_value=True))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='gtn_noise_std', lower=0.01, upper=1, log=True, default_value=0.1))

        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='ddqn_init_episodes', lower=1, upper=20, log=True, default_value=10))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='ddqn_batch_size', lower=64, upper=256, log=False, default_value=128))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='ddqn_gamma', lower=0.001, upper=0.1, log=True, default_value=0.01))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='ddqn_lr', lower=1e-4, upper=5e-3, log=True, default_value=1e-3))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='ddqn_tau', lower=0.005, upper=0.05, log=True, default_value=0.01))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='ddqn_eps_init', lower=0.8, upper=1, log=True, default_value=0.9))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='ddqn_eps_min', lower=0.005, upper=0.05, log=True, default_value=0.05))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='ddqn_eps_decay', lower=0.01, upper=0.2, log=True, default_value=0.1))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='ddqn_activation_fn', choices=['tanh', 'relu', 'leakyrelu', 'prelu'], default_value='relu'))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='ddqn_hidden_size', lower=48, upper=192, log=True, default_value=128))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='ddqn_hidden_layer', lower=1, upper=2, log=False, default_value=2))

        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='cartpole_activation_fn', choices=['tanh', 'relu', 'leakyrelu', 'prelu'], default_value='leakyrelu'))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='cartpole_hidden_size', lower=48, upper=192, log=True, default_value=128))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='cartpole_hidden_layer', lower=1, upper=2, log=False, default_value=1))

        return cs


    def get_specific_config(self, cso, default_config, budget):
        config = deepcopy(default_config)

        config["agents"]['gtn']['score_transform_type'] = cso["gtn_score_transform_type"]
        config["agents"]['gtn']['step_size'] = cso["gtn_step_size"]
        config["agents"]['gtn']['mirrored_sampling'] = cso["gtn_mirrored_sampling"]
        config["agents"]['gtn']['noise_std'] = cso["gtn_noise_std"]

        config["agents"]['ddqn']['init_episodes'] = cso["ddqn_init_episodes"]
        config["agents"]['ddqn']['batch_size'] = cso["ddqn_batch_size"]
        config["agents"]['ddqn']['gamma'] = 1-cso["ddqn_gamma"]
        config["agents"]['ddqn']['lr'] = cso["ddqn_lr"]
        config["agents"]['ddqn']['tau'] = cso["ddqn_tau"]
        config["agents"]['ddqn']['eps_init'] = cso["ddqn_eps_init"]
        config["agents"]['ddqn']['eps_min'] = cso["ddqn_eps_min"]
        config["agents"]['ddqn']['eps_decay'] = 1-cso["ddqn_eps_decay"]
        config["agents"]['ddqn']['activation_fn'] = cso["ddqn_activation_fn"]
        config["agents"]['ddqn']['hidden_size'] = cso["ddqn_hidden_size"]
        config["agents"]['ddqn']['hidden_layer'] = cso["ddqn_hidden_layer"]

        config["envs"]['CartPole-v0']['activation_fn'] = cso["cartpole_activation_fn"]
        config["envs"]['CartPole-v0']['hidden_size'] = cso["cartpole_hidden_size"]
        config["envs"]['CartPole-v0']['hidden_layer'] = cso["cartpole_hidden_layer"]

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

        try:
            score = 0
            for _ in range(3):
                gtn = GTN_Master(config, bohb_id=bohb_id)
                _, score_list = gtn.run()
                score += len(score_list)
            error = ""
        except:
            score = float('Inf')
            score_list = []
            error = traceback.format_exc()
            print(error)

        info = {}
        info['error'] = str(error)
        info['config'] = str(config)
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
