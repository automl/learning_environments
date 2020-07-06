import datetime
import sys
import yaml
import random
import numpy as np
import torch
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from copy import deepcopy
from agents.match_env import MatchEnv
from envs.env_factory import EnvFactory
from automl.bohb_optim import run_bohb_parallel, run_bohb_serial


class ExperimentWrapper():
    def get_bohb_parameters(self):
        params = {}
        params['seed'] = 42
        params['min_budget'] = 1
        params['max_budget'] = 8
        params['eta'] = 2
        params['iterations'] = 1000

        return params


    def get_configspace(self):
        cs = CS.ConfigurationSpace()

        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='oversample', lower=1, upper=2, log=True, default_value=1.1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='lr', lower=1e-5, upper=1e-2, log=True, default_value=1e-3))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='weight_decay', lower=1e-10, upper=1e-1, log=True, default_value=1e-3))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='batch_size', lower=32, upper=256, log=True, default_value=128))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='early_out_diff', lower=1e-5, upper=1e-1, log=True, default_value=0.01))
        #cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='early_out_diff', lower=9e-1, upper=1e0, log=True, default_value=0.91))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='early_out_num', lower=10, upper=200, log=False, default_value=50))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='steps', lower=100, upper=20000, log=False, default_value=5000))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='step_size', lower=100, upper=2000, log=True, default_value=1000))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='gamma', lower=0.1, upper=0.9, log=False, default_value=0.5))

        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='activation_fn', choices=['tanh', 'relu', 'leakyrelu', 'prelu'], default_value='relu'))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='hidden_size', lower=32, upper=1024, log=True, default_value=224))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='hidden_layer', lower=0, upper=2, log=False, default_value=1))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='zero_init', choices=[False, True], default_value=False))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='weight_norm', choices=[False, True], default_value=True))

        return cs


    def get_specific_config(self, cso, default_config, budget):
        config = deepcopy(default_config)

        config["env_name"] = 'Pendulum-v0'

        config["agents"]['match_env']['oversample'] = cso["oversample"]
        config["agents"]['match_env']['lr'] = cso["lr"]
        config["agents"]['match_env']['weight_decay'] = cso["weight_decay"]
        config["agents"]['match_env']['batch_size'] = cso["batch_size"]
        config["agents"]['match_env']['early_out_diff'] = cso["early_out_diff"]
        config["agents"]['match_env']['early_out_num'] = cso["early_out_num"]
        config["agents"]['match_env']['steps'] = cso["steps"]
        config["agents"]['match_env']['step_size'] = cso["step_size"]
        config["agents"]['match_env']['gamma'] = cso["gamma"]

        config["envs"]['Pendulum-v0']['activation_fn'] = cso["activation_fn"]
        config["envs"]['Pendulum-v0']['hidden_size'] = cso["hidden_size"]
        config["envs"]['Pendulum-v0']['hidden_layer'] = cso["hidden_layer"]
        config["envs"]['Pendulum-v0']['zero_init'] = cso["zero_init"]
        config["envs"]['Pendulum-v0']['weight_norm'] = cso["weight_norm"]

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
            # generate environment
            env_fac = EnvFactory(config)
            real_env = env_fac.generate_default_real_env()
            virtual_env = env_fac.generate_default_virtual_env()

            match_env = MatchEnv(config = config)
            match_env.train(real_env = real_env,
                            virtual_env = virtual_env,
                            input_seed = 0)
            loss, diff_state, diff_reward, diff_done = \
                match_env.validate(real_env = real_env,
                                   virtual_env = virtual_env,
                                   input_seed = 0,
                                   validate_samples = 10000)
            diff_state = float(diff_state.cpu().data.numpy())
            diff_reward = float(diff_reward.cpu().data.numpy())
            diff_done = float(diff_done.cpu().data.numpy())
            score = diff_state + diff_reward + diff_done
        except:
            score = float('Inf')
            diff_state = float('Inf')
            diff_reward = float('Inf')
            diff_done = float('Inf')

        info['config'] = str(config)
        info['diff_state'] = diff_state
        info['diff_reward'] = diff_reward
        info['diff_done'] = diff_done

        print('----------------------------')
        print('FINAL SCORE: ' + str(score))
        print('DIFF: ' + str(diff_state) + ' ' + str(diff_reward) + ' ' + str(diff_done))
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
    run_id = 'optimize_match_env_' + x.strftime("%Y-%m-%d-%H")

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
