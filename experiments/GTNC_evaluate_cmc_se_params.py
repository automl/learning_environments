import datetime
import time
import sys
import traceback
import math
import yaml
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import random
import numpy as np
import torch
import statistics
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
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='gtn_noise_std', lower=0.001, upper=1, log=True, default_value=0.01))

        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='td3_init_episodes', lower=1, upper=100, log=True, default_value=20))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='td3_batch_size', lower=32, upper=256, log=False, default_value=128))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_gamma', lower=0.001, upper=0.1, log=True, default_value=0.01))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_lr', lower=1e-5, upper=5e-2, log=True, default_value=1e-3))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_tau', lower=0.005, upper=0.05, log=True, default_value=0.01))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='td3_policy_delay', lower=1, upper=3, log=False, default_value=2))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='td3_activation_fn', choices=['tanh', 'relu', 'leakyrelu', 'prelu'], default_value='relu'))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='td3_hidden_size', lower=48, upper=192, log=True, default_value=128))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='td3_hidden_layer', lower=1, upper=2, log=False, default_value=2))
        # cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='td3_same_action_num', lower=2, upper=3, log=False, default_value=2))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_action_std', lower=0.05, upper=0.6, log=True, default_value=0.1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_policy_std', lower=0.1, upper=0.6, log=True, default_value=0.2))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_policy_std_clip', lower=0.25, upper=1, log=True, default_value=0.5))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_early_out_virtual_diff', lower=1e-2, upper=1e-1, log=True, default_value=3e-2))

        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='cmc_activation_fn', choices=['tanh', 'relu', 'leakyrelu', 'prelu'], default_value='leakyrelu'))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='cmc_hidden_size', lower=48, upper=192, log=True, default_value=128))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='cmc_hidden_layer', lower=1, upper=3, log=False, default_value=1))

        return cs


    def get_specific_config(self, cso, default_config, budget):
        config = deepcopy(default_config)

        config["agents"]['gtn']['score_transform_type'] = cso["gtn_score_transform_type"]
        config["agents"]['gtn']['step_size'] = cso["gtn_step_size"]
        config["agents"]['gtn']['mirrored_sampling'] = cso["gtn_mirrored_sampling"]
        config["agents"]['gtn']['noise_std'] = cso["gtn_noise_std"]

        config["agents"]['td3']['init_episodes'] = cso["td3_init_episodes"]
        config["agents"]['td3']['batch_size'] = cso["td3_batch_size"]
        config["agents"]['td3']['gamma'] = 1-cso["td3_gamma"]
        config["agents"]['td3']['lr'] = cso["td3_lr"]
        config["agents"]['td3']['tau'] = cso["td3_tau"]
        config["agents"]['td3']['policy_delay'] = cso["td3_policy_delay"]
        config["agents"]['td3']['activation_fn'] = cso["td3_activation_fn"]
        config["agents"]['td3']['hidden_size'] = cso["td3_hidden_size"]
        config["agents"]['td3']['hidden_layer'] = cso["td3_hidden_layer"]
        # config["agents"]['td3']['same_action_num'] = cso["td3_same_action_num"]
        config["agents"]['td3']['action_std'] = cso["td3_action_std"]
        config["agents"]['td3']['policy_std'] = cso["td3_policy_std"]
        config["agents"]['td3']['policy_std_clip'] = cso["td3_policy_std_clip"]
        config["agents"]['td3']['early_out_virtual_diff'] = cso["td3_early_out_virtual_diff"]

        config["envs"]['MountainCarContinuous-v0']['activation_fn'] = cso["cmc_activation_fn"]
        config["envs"]['MountainCarContinuous-v0']['hidden_size'] = cso["cmc_hidden_size"]
        config["envs"]['MountainCarContinuous-v0']['hidden_layer'] = cso["cmc_hidden_layer"]

        global reward_env_type
        config["agents"]['gtn']['synthetic_env_type'] = 0

        return config


    def compute(self, working_dir, bohb_id, config_id, cso, budget, *args, **kwargs):
        with open("default_config_cmc.yaml", 'r') as stream:
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
            score = -sorted(score_list)[-1]
            if math.isnan(score):
                score = float('Inf')
            error = ""
        except:
            score = float('Inf')
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

    id = int(sys.argv[1])
    bohb_workers = int(sys.argv[2])

    run_id = 'SE_evaluate_cmc_se_params_' + x.strftime("%Y-%m-%d-%H")

    seed = id+int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    res = run_bohb_parallel(id=id,
                            bohb_workers=bohb_workers,
                            run_id=run_id,
                            experiment_wrapper=ExperimentWrapper())
