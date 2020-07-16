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

        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='gtn_max_iterations', lower=1, upper=10, log=False, default_value=5))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='gtn_type_0', lower=1, upper=3, log=False, default_value=1))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='gtn_type_1', lower=1, upper=3, log=False, default_value=1))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='gtn_type_2', lower=1, upper=3, log=False, default_value=1))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='gtn_type_3', lower=1, upper=3, log=False, default_value=1))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='gtn_type_4', lower=1, upper=3, log=False, default_value=1))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='gtn_type_5', lower=1, upper=3, log=False, default_value=1))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='gtn_type_6', lower=1, upper=3, log=False, default_value=1))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='gtn_type_7', lower=1, upper=3, log=False, default_value=1))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='gtn_type_8', lower=1, upper=3, log=False, default_value=1))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='gtn_type_9', lower=1, upper=3, log=False, default_value=1))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='gtn_different_envs', lower=1, upper=10, log=False, default_value=5))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='gtn_match_step_size', lower=0.05, upper=1, log=True, default_value=0.1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='gtn_real_step_size', lower=0.05, upper=1, log=True, default_value=0.1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='gtn_virtual_step_size', lower=0.05, upper=1, log=True, default_value=0.1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='gtn_both_step_size', lower=0.05, upper=1, log=True, default_value=0.1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='gtn_input_seed_mean', lower=0.1, upper=5, log=True, default_value=0.1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='gtn_input_seed_range', lower=0.001, upper=5, log=True, default_value=0.1))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='gtn_pretrain_agent', choices=[False, True], default_value=True))

        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='em_oversampling', lower=1, upper=2, log=True, default_value=1.5))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='em_lr', lower=1e-3, upper=1e-1, log=True, default_value=2e-3))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='em_weight_decay', lower=1e-12, upper=1e-6, log=True, default_value=1e-9))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='em_batch_size', lower=64, upper=512, log=True, default_value=128))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='em_early_out_diff', lower=1e-7, upper=1e-3, log=True, default_value=1e-4))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='em_max_steps', lower=1, upper=10000, log=True, default_value=5000))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='em_step_size', lower=200, upper=2000, log=True, default_value=1000))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='em_gamma', lower=0.6, upper=1, log=False, default_value=0.7))

        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_lr', lower=3e-4, upper=1e-2, log=True, default_value=1e-3))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='td3_hidden_size', lower=128, upper=1024, log=True, default_value=256))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='td3_hidden_layer', lower=1, upper=2, log=False, default_value=1))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='td3_weight_norm', choices=[False, True], default_value=False))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_weight_decay', lower=1e-12, upper=1e-6, log=True, default_value=1e-9))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='td3_optim_env_with_actor', choices=[False, True], default_value=False))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='td3_optim_env_with_critic', choices=[False, True], default_value=False))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_match_weight_actor', lower=1e0, upper=1e8, log=True, default_value=1e5))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_match_weight_critic', lower=1e0, upper=1e8, log=True, default_value=1e5))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='td3_match_batch_size', lower=64, upper=512, log=True, default_value=256))

        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='pen_hidden_size', lower=128, upper=2024, log=True, default_value=224))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='pen_hidden_layer', lower=1, upper=3, log=True, default_value=2))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='pen_input_seed_dim', lower=1, upper=64, log=True, default_value=4))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='pen_input_seed_mean', lower=0.01, upper=10, log=True, default_value=1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='pen_input_seed_range', lower=0.01, upper=10, log=True, default_value=1))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='pen_weight_norm', choices=[False, True], default_value=True))

        return cs


    def get_specific_config(self, cso, default_config, budget):
        config = deepcopy(default_config)

        config["env_name"] = 'Pendulum-v0'

        config["agents"]['gtn']['max_iterations'] = cso["gtn_max_iterations"]
        config["agents"]['gtn']['type_0'] = cso["gtn_type_0"]
        config["agents"]['gtn']['type_1'] = cso["gtn_type_1"]
        config["agents"]['gtn']['type_2'] = cso["gtn_type_2"]
        config["agents"]['gtn']['type_3'] = cso["gtn_type_3"]
        config["agents"]['gtn']['type_4'] = cso["gtn_type_4"]
        config["agents"]['gtn']['type_5'] = cso["gtn_type_5"]
        config["agents"]['gtn']['type_6'] = cso["gtn_type_6"]
        config["agents"]['gtn']['type_7'] = cso["gtn_type_7"]
        config["agents"]['gtn']['type_8'] = cso["gtn_type_8"]
        config["agents"]['gtn']['type_9'] = cso["gtn_type_9"]
        config["agents"]['gtn']['different_envs'] = cso["gtn_different_envs"]
        config["agents"]['gtn']['match_step_size'] = cso["gtn_match_step_size"]
        config["agents"]['gtn']['real_step_size'] = cso["gtn_real_step_size"]
        config["agents"]['gtn']['virtual_step_size'] = cso["gtn_virtual_step_size"]
        config["agents"]['gtn']['both_step_size'] = cso["gtn_both_step_size"]
        config["agents"]['gtn']['input_seed_mean'] = cso["gtn_input_seed_mean"]
        config["agents"]['gtn']['input_seed_range'] = cso["gtn_input_seed_range"]
        config["agents"]['gtn']['pretrain_agent'] = cso["gtn_pretrain_agent"]

        config["agents"]['env_matcher']['oversampling'] = cso["em_oversampling"]
        config["agents"]['env_matcher']['lr'] = cso["em_lr"]
        config["agents"]['env_matcher']['weight_decay'] = cso["em_weight_decay"]
        config["agents"]['env_matcher']['batch_size'] = cso["em_batch_size"]
        config["agents"]['env_matcher']['early_out_diff'] = cso["em_early_out_diff"]
        config["agents"]['env_matcher']['max_steps'] = cso["em_max_steps"]
        config["agents"]['env_matcher']['step_size'] = cso["em_step_size"]
        config["agents"]['env_matcher']['gamma'] = cso["em_gamma"]

        config["agents"]["td3"]["lr"] = cso["td3_lr"]
        config["agents"]["td3"]["hidden_size"] = cso["td3_hidden_size"]
        config["agents"]["td3"]["hidden_layer"] = cso["td3_hidden_layer"]
        config["agents"]["td3"]["weight_norm"] = cso["td3_weight_norm"]
        config["agents"]["td3"]["weight_decay"] = cso["td3_weight_decay"]
        config["agents"]["td3"]["optim_env_with_actor"] = cso["td3_optim_env_with_actor"]
        config["agents"]["td3"]["optim_env_with_critic"] = cso["td3_optim_env_with_critic"]
        config["agents"]["td3"]["match_weight_actor"] = cso["td3_match_weight_actor"]
        config["agents"]["td3"]["match_weight_critic"] = cso["td3_match_weight_critic"]
        config["agents"]["td3"]["match_batch_size"] = cso["td3_match_batch_size"]

        config["envs"]['Pendulum-v0']['hidden_size'] = cso["pen_hidden_size"]
        config["envs"]['Pendulum-v0']['hidden_layer'] = cso["pen_hidden_layer"]
        config["envs"]['Pendulum-v0']['input_seed_dim'] = cso["pen_input_seed_dim"]
        config["envs"]['Pendulum-v0']['input_seed_mean'] = cso["pen_input_seed_mean"]
        config["envs"]['Pendulum-v0']['input_seed_range'] = cso["pen_input_seed_range"]
        config["envs"]['Pendulum-v0']['weight_norm'] = cso["pen_weight_norm"]

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
        error = ""
        try:
            gtn = GTN(config)
            order = gtn.train()
            score = gtn.test()
        except:
            score = float('Inf')
            error = traceback.format_exc()
            print(error)

        info['config'] = str(config)
        info['order'] = str(order)
        info['error'] = str(error)

        print('----------------------------')
        print('FINAL SCORE: ' + str(score))
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
    run_id = 'GTN_params_reduced_bohb_' + x.strftime("%Y-%m-%d-%H")

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
