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

        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='gtn_max_iterations', lower=0, upper=5, log=False, default_value=5))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='gtn_type_0', lower=1, upper=2, log=False, default_value=1))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='gtn_type_1', lower=1, upper=2, log=False, default_value=1))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='gtn_type_2', lower=1, upper=2, log=False, default_value=1))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='gtn_type_3', lower=1, upper=2, log=False, default_value=1))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='gtn_type_4', lower=1, upper=2, log=False, default_value=1))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='gtn_different_envs', lower=1, upper=4, log=False, default_value=1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='gtn_real_step_size', lower=0.1, upper=1, log=True, default_value=0.2))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='gtn_virtual_step_size', lower=0.1, upper=1, log=True, default_value=0.2))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='gtn_pretrain_agent', choices=[False, True], default_value=True))

        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='em_oversampling', lower=1, upper=1.3, log=True, default_value=1.1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='em_lr', lower=1e-4, upper=1e-2, log=True, default_value=1e-4))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='em_batch_size', lower=128, upper=1024, log=True, default_value=256))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='em_max_steps', lower=1, upper=10000, log=True, default_value=5000))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='em_step_size', lower=100, upper=1000, log=True, default_value=500))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='em_match_loss_state', choices=['L1', 'L2'], default_value='L1'))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='em_match_loss_reward', choices=['L1', 'L2'], default_value='L2'))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='em_match_loss_done', choices=['L1', 'L2'], default_value='L1'))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='em_variation_type', lower=1, upper=3, log=True, default_value=1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='em_variation_weight', lower=1e-4, upper=1e-1, log=True, default_value=1e-3))

        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_lr', lower=1e-4, upper=5e-3, log=True, default_value=1e-3))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_gamma', lower=1e-3, upper=2e-2, log=True, default_value=1e-2))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_tau', lower=0.002, upper=0.05, log=True, default_value=0.01))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='td3_hidden_size', lower=64, upper=512, log=True, default_value=256))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='td3_hidden_layer', lower=1, upper=2, log=False, default_value=1))

        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='env_hidden_size', lower=64, upper=512, log=True, default_value=256))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='env_hidden_layer', lower=1, upper=2, log=True, default_value=1))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='env_input_seed_dim', lower=1, upper=4, log=True, default_value=4))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='env_input_seed_mean', lower=0.01, upper=10, log=True, default_value=0.1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='env_input_seed_range', lower=0.01, upper=10, log=True, default_value=0.1))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='env_weight_norm', choices=[False, True], default_value=False))

        return cs

    def get_specific_config(self, cso, default_config, budget):
        config = deepcopy(default_config)

        env_name = config["env_name"]
        config["render_env"] = False

        config["agents"]['gtn']['max_iterations'] = cso["gtn_max_iterations"]
        config["agents"]['gtn']['type_0'] = cso["gtn_type_0"]
        config["agents"]['gtn']['type_1'] = cso["gtn_type_1"]
        config["agents"]['gtn']['type_2'] = cso["gtn_type_2"]
        config["agents"]['gtn']['type_3'] = cso["gtn_type_3"]
        config["agents"]['gtn']['type_4'] = cso["gtn_type_4"]
        config["agents"]['gtn']['different_envs'] = cso["gtn_different_envs"]
        config["agents"]['gtn']['real_step_size'] = cso["gtn_real_step_size"]
        config["agents"]['gtn']['virtual_step_size'] = cso["gtn_virtual_step_size"]
        config["agents"]['gtn']['pretrain_agent'] = cso["gtn_pretrain_agent"]

        config["agents"]['env_matcher']['oversampling'] = cso["em_oversampling"]
        config["agents"]['env_matcher']['lr'] = cso["em_lr"]
        config["agents"]['env_matcher']['batch_size'] = cso["em_batch_size"]
        config["agents"]['env_matcher']['max_steps'] = cso["em_max_steps"]-1
        config["agents"]['env_matcher']['step_size'] = cso["em_step_size"]
        config["agents"]['env_matcher']['match_loss_state'] = cso["em_match_loss_state"]
        config["agents"]['env_matcher']['match_loss_reward'] = cso["em_match_loss_reward"]
        config["agents"]['env_matcher']['match_loss_done'] = cso["em_match_loss_done"]
        config["agents"]['env_matcher']['variation_type'] = cso["em_variation_type"]
        config["agents"]['env_matcher']['variation_weight'] = cso["em_variation_weight"]

        config["agents"]["td3"]["lr"] = cso["td3_lr"]
        config["agents"]["td3"]["gamma"] = 1-cso["td3_gamma"]
        config["agents"]["td3"]["tau"] = cso["td3_tau"]
        config["agents"]["td3"]["hidden_size"] = cso["td3_hidden_size"]
        config["agents"]["td3"]["hidden_layer"] = cso["td3_hidden_layer"]

        config["envs"][env_name]['hidden_size'] = cso["env_hidden_size"]
        config["envs"][env_name]['hidden_layer'] = cso["env_hidden_layer"]
        config["envs"][env_name]['input_seed_dim'] = cso["env_input_seed_dim"]
        config["envs"][env_name]['input_seed_mean'] = cso["env_input_seed_mean"]
        config["envs"][env_name]['input_seed_range'] = cso["env_input_seed_range"]
        config["envs"][env_name]['weight_norm'] = cso["env_weight_norm"]

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
            order, timings = gtn.train()
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
