import datetime
import sys
import yaml
import random
import numpy as np
import torch
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from copy import deepcopy
from agents.TD3 import TD3
from agents.match_env import MatchEnv
from agents.REPTILE import reptile_train_agent
from envs.env_factory import EnvFactory
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

        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='me_oversample', lower=1, upper=3, log=True, default_value=1.1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='me_lr', lower=1e-4, upper=1e-1, log=True, default_value=1e-3))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='me_weight_decay', lower=1e-12, upper=1e-3, log=True, default_value=1e-3))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='me_batch_size', lower=64, upper=512, log=True, default_value=128))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='me_early_out_diff', lower=1e-7, upper=1e-3, log=True, default_value=1e-4))
        #cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='early_out_diff', lower=9e-1, upper=1e0, log=True, default_value=0.91))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='me_early_out_num', lower=10, upper=1000, log=False, default_value=100))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='me_steps', lower=200, upper=20000, log=False, default_value=5000))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='me_step_size', lower=200, upper=2000, log=True, default_value=1000))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='me_gamma', lower=0.3, upper=1, log=False, default_value=0.7))

        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='rep_step_size', lower=0.01, upper=1, log=False, default_value=0.1))

        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='pen_activation_fn', choices=['tanh', 'relu', 'leakyrelu', 'prelu'], default_value='relu'))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='pen_hidden_size', lower=64, upper=1024, log=True, default_value=224))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='pen_hidden_layer', lower=1, upper=2, log=False, default_value=1))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='pen_zero_init', choices=[False, True], default_value=False))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='pen_weight_norm', choices=[False, True], default_value=True))

        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='td3_init_episodes', lower=1, upper=20, log=True, default_value=10))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='td3_batch_size', lower=64, upper=512, log=True, default_value=256))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_gamma', lower=0.001, upper=0.1, log=True, default_value=0.01))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_lr', lower=1e-4, upper=1e-1, log=True, default_value=3e-4))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_weight_decay', lower=1e-12, upper=1e-4, log=True, default_value=1e-10))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_tau', lower=0.001, upper=0.1, log=True, default_value=0.01))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='td3_policy_delay', lower=1, upper=5, log=False, default_value=2))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='td3_rb_size', lower=1000, upper=1000000, log=True, default_value=100000))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='td3_hidden_size', lower=64, upper=512, log=True, default_value=224))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='td3_activation_fn', choices=['relu', 'tanh', 'leakyrelu', 'prelu'], default_value='relu'))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='td3_weight_norm', choices=[False, True], default_value=True))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_action_std', lower=0.01, upper=10, log=True, default_value=0.1))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='td3_early_out_num', lower=1, upper=5, log=True, default_value=3))

        return cs


    def get_specific_config(self, cso, default_config, budget):
        config = deepcopy(default_config)

        config["env_name"] = 'Pendulum-v0'

        config["agents"]['match_env']['oversample'] = cso["me_oversample"]
        config["agents"]['match_env']['lr'] = cso["me_lr"]
        config["agents"]['match_env']['weight_decay'] = cso["me_weight_decay"]
        config["agents"]['match_env']['batch_size'] = cso["me_batch_size"]
        config["agents"]['match_env']['early_out_diff'] = cso["me_early_out_diff"]
        config["agents"]['match_env']['early_out_num'] = cso["me_early_out_num"]
        config["agents"]['match_env']['steps'] = cso["me_steps"]
        config["agents"]['match_env']['step_size'] = cso["me_step_size"]
        config["agents"]['match_env']['gamma'] = cso["me_gamma"]

        config["envs"]['Pendulum-v0']['activation_fn'] = cso["pen_activation_fn"]
        config["envs"]['Pendulum-v0']['hidden_size'] = cso["pen_hidden_size"]
        config["envs"]['Pendulum-v0']['hidden_layer'] = cso["pen_hidden_layer"]
        config["envs"]['Pendulum-v0']['zero_init'] = cso["pen_zero_init"]
        config["envs"]['Pendulum-v0']['weight_norm'] = cso["pen_weight_norm"]

        config["agents"]["reptile"]["step_size"] = cso["rep_step_size"]

        config["agents"]["td3"]["init_episodes"] = cso["td3_init_episodes"]
        config["agents"]["td3"]["batch_size"] = cso["td3_batch_size"]
        config["agents"]["td3"]["gamma"] = 1-cso["td3_gamma"]
        config["agents"]["td3"]["lr"] = cso["td3_lr"]
        config["agents"]["td3"]["weight_decay"] = cso["td3_weight_decay"]
        config["agents"]["td3"]["tau"] = cso["td3_tau"]
        config["agents"]["td3"]["policy_delay"] = cso["td3_policy_delay"]
        config["agents"]["td3"]["rb_size"] = cso["td3_rb_size"]
        config["agents"]["td3"]["hidden_size"] = cso["td3_hidden_size"]
        config["agents"]["td3"]["activation_fn"] = cso["td3_activation_fn"]
        config["agents"]["td3"]["weight_norm"] = cso["td3_weight_norm"]
        config["agents"]["td3"]["action_std"] = cso["td3_action_std"]
        config["agents"]["td3"]["early_out_num"] = cso["td3_early_out_num"]

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

        try:
            env_fac = EnvFactory(config)
            real_env = env_fac.generate_default_real_env()
            virtual_env = env_fac.generate_default_virtual_env()

            td3 = TD3(state_dim=real_env.get_state_dim(),
                      action_dim=real_env.get_action_dim(),
                      config=config)

            # first match
            print("-- matching virtual env to real env --")
            match_env = MatchEnv(config=config)
            match_env.train(real_env=real_env,
                            virtual_env=virtual_env,
                            input_seed=0)

            # then train on virtual env
            print("-- training on virtual env --")
            reptile_train_agent(agent=td3,
                                env=virtual_env,
                                step_size=config["agents"]["reptile"]["step_size"])

            # then train on real env
            # ideally the reptile update works and we can train on this environment rather quickly
            print("-- training on real env --")
            reward_list = reptile_train_agent(agent=td3,
                                              env=real_env,
                                              step_size=1)
            score = len(reward_list)

        except:
            score = float('Inf')

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
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    x = datetime.datetime.now()
    run_id = 'evaluate_match_bohb_' + x.strftime("%Y-%m-%d-%H")

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
