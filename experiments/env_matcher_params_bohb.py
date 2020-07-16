import datetime
import traceback
import sys
import yaml
import random
import numpy as np
import torch
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from copy import deepcopy
from agents.env_matcher import EnvMatcher
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

        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='oversampling', lower=1, upper=3, log=True, default_value=1.1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='lr', lower=1e-4, upper=1e-1, log=True, default_value=1e-3))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='weight_decay', lower=1e-12, upper=1e-3, log=True, default_value=1e-3))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='batch_size', lower=64, upper=512, log=True, default_value=128))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='early_out_diff', lower=1e-7, upper=1e-3, log=True, default_value=1e-4))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='early_out_num', lower=10, upper=1000, log=False, default_value=100))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='steps', lower=200, upper=10000, log=False, default_value=5000))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='step_size', lower=200, upper=2000, log=True, default_value=1000))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='gamma', lower=0.3, upper=1, log=False, default_value=0.7))

        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='activation_fn', choices=['tanh', 'relu', 'leakyrelu', 'prelu'], default_value='relu'))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='hidden_size', lower=64, upper=1024, log=True, default_value=224))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='hidden_layer', lower=1, upper=2, log=False, default_value=1))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='input_seed_dim', lower=1, upper=32, log=True, default_value=4))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='input_seed_mean', lower=0.01, upper=10, log=True, default_value=1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='input_seed_range', lower=0.01, upper=10, log=True, default_value=1))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='zero_init', choices=[False, True], default_value=False))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='weight_norm', choices=[False, True], default_value=True))

        return cs


    def get_specific_config(self, cso, default_config, budget):
        config = deepcopy(default_config)

        config["env_name"] = 'Pendulum-v0'

        config["agents"]['env_matcher']['oversampling'] = cso["oversampling"]
        config["agents"]['env_matcher']['lr'] = cso["lr"]
        config["agents"]['env_matcher']['weight_decay'] = cso["weight_decay"]
        config["agents"]['env_matcher']['batch_size'] = cso["batch_size"]
        config["agents"]['env_matcher']['early_out_diff'] = cso["early_out_diff"]
        config["agents"]['env_matcher']['early_out_num'] = cso["early_out_num"]
        config["agents"]['env_matcher']['steps'] = cso["steps"]
        config["agents"]['env_matcher']['step_size'] = cso["step_size"]
        config["agents"]['env_matcher']['gamma'] = cso["gamma"]

        config["envs"]['Pendulum-v0']['activation_fn'] = cso["activation_fn"]
        config["envs"]['Pendulum-v0']['hidden_size'] = cso["hidden_size"]
        config["envs"]['Pendulum-v0']['hidden_layer'] = cso["hidden_layer"]
        config["envs"]['Pendulum-v0']['input_seed_dim'] = cso["input_seed_dim"]
        config["envs"]['Pendulum-v0']['input_seed_mean'] = cso["input_seed_mean"]
        config["envs"]['Pendulum-v0']['input_seed_range'] = cso["input_seed_range"]
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
            real_envs = []
            input_seeds = []
            for i in range(10):
                real_envs.append(env_fac.generate_random_real_env())
                input_seeds.append(env_fac.generate_random_input_seed())
            virtual_env = env_fac.generate_default_virtual_env()

            env_matcher = EnvMatcher(config = config)
            env_matcher.train(real_envs = real_envs,
                              virtual_env = virtual_env,
                              input_seeds = input_seeds)
            loss, diff_state, diff_reward, diff_done = \
                env_matcher.test(real_envs = real_envs,
                                 virtual_env = virtual_env,
                                 input_seeds = input_seeds,
                                 oversampling = 1.5,
                                 test_samples = 10000)
            diff_state = float(diff_state.cpu().data.numpy())
            diff_reward = float(diff_reward.cpu().data.numpy())
            diff_done = float(diff_done.cpu().data.numpy())
            score = diff_state + diff_reward + diff_done
            error = ''
        except:
            score = float('Inf')
            diff_state = float('Inf')
            diff_reward = float('Inf')
            diff_done = float('Inf')
            error = traceback.format_exc()

        info['config'] = str(config)
        info['diff_state'] = diff_state
        info['diff_reward'] = diff_reward
        info['diff_done'] = diff_done
        info['error'] = str(error)

        print('----------------------------')
        print('FINAL SCORE: ' + str(score))
        print('DIFF: ' + str(diff_state) + ' ' + str(diff_reward) + ' ' + str(diff_done))
        print('ERR: ' + str(error))
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
    run_id = 'env_matcher_params_bohb_' + x.strftime("%Y-%m-%d-%H")

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
