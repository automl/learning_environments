import datetime
import sys
import yaml
import random
import numpy as np
import statistics
import torch
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from copy import deepcopy
from agents.TD3 import TD3
from envs.env_factory import EnvFactory
from automl.bohb_optim import run_bohb_parallel, run_bohb_serial

NUM_EVALS = 3

class ExperimentWrapper():
    def get_bohb_parameters(self):
        params = {}
        params['min_budget'] = 1
        params['max_budget'] = 8
        params['eta'] = 2
        params['iterations'] = 1000
        params['random_fraction'] = 0.3

        return params


    def get_configspace(self):
        cs = CS.ConfigurationSpace()

        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='td3_batch_size', lower=64, upper=256, log=False, default_value=128))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_gamma', lower=0.001, upper=0.1, log=True, default_value=0.01))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_lr', lower=1e-4, upper=5e-3, log=True, default_value=1e-3))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_tau', lower=0.005, upper=0.05, log=True, default_value=0.01))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='td3_policy_delay', lower=1, upper=3, log=False, default_value=2))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='td3_activation_fn', choices=['tanh', 'relu', 'leakyrelu', 'prelu'], default_value='relu'))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='td3_hidden_size', lower=48, upper=192, log=True, default_value=128))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='td3_hidden_layer', lower=1, upper=2, log=False, default_value=2))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='td3_same_action_num', lower=1, upper=3, log=False, default_value=1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_action_std', lower=0.05, upper=0.2, log=True, default_value=0.1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_policy_std', lower=0.1, upper=0.4, log=True, default_value=0.2))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_policy_std_clip', lower=0.25, upper=1, log=True, default_value=0.5))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_early_out_virtual_diff', lower=1e-2, upper=1e-1, log=True, default_value=3e-2))

        return cs


    def get_specific_config(self, cso, default_config, budget):
        config = deepcopy(default_config)

        config["agents"]['td3']['batch_size'] = cso["td3_batch_size"]
        config["agents"]['td3']['gamma'] = 1-cso["td3_gamma"]
        config["agents"]['td3']['lr'] = cso["td3_lr"]
        config["agents"]['td3']['tau'] = cso["td3_tau"]
        config["agents"]['td3']['policy_delay'] = cso["td3_policy_delay"]
        config["agents"]['td3']['activation_fn'] = cso["td3_activation_fn"]
        config["agents"]['td3']['hidden_size'] = cso["td3_hidden_size"]
        config["agents"]['td3']['hidden_layer'] = cso["td3_hidden_layer"]
        config["agents"]['td3']['same_action_num'] = cso["td3_same_action_num"]
        config["agents"]['td3']['action_std'] = cso["td3_action_std"]
        config["agents"]['td3']['policy_std'] = cso["td3_policy_std"]
        config["agents"]['td3']['policy_std_clip'] = cso["td3_policy_std_clip"]
        config["agents"]['td3']['early_out_virtual_diff'] = cso["td3_early_out_virtual_diff"]
        #config["device"] = 'cuda'

        return config


    def compute(self, working_dir, bohb_id, config_id, cso, budget, *args, **kwargs):
        with open("default_config_pendulum_td3_opt_2.yaml", 'r') as stream:
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

        real_env = env_fac.generate_real_env()
        reward_env = env_fac.generate_reward_env()
        save_dict = torch.load('/home/fr/fr_fr/fr_tn87/master_thesis/learning_environments/results/GTNC_evaluate_pendulum_params_2020-12-24-13_2/GTN_models_Pendulum-v0/Pendulum-v0_IAMD5O.pt')
        #save_dict = torch.load('/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_pendulum_params_2020-12-24-13_2/GTN_models_Pendulum-v0/Pendulum-v0_IAMD5O.pt')
        #save_dict = torch.load('/home/dingsda/master_thesis/learning_environments/results/GTNC_evaluate_pendulum_params_2020-12-24-13_2/GTN_models_Pendulum-v0/Pendulum-v0_IAMD5O.pt')
        #config = save_dict['config']
        # TODO: dirty hack for compatibility
        save_dict['model']['env.reward_net.5.weight'] = save_dict['model']['env.reward_net.4.weight']
        save_dict['model']['env.reward_net.5.bias'] = save_dict['model']['env.reward_net.4.bias']
        del save_dict['model']['env.reward_net.4.weight']
        del save_dict['model']['env.reward_net.4.bias']
        reward_env.load_state_dict(save_dict['model'])

        score = 0
        for i in range(NUM_EVALS):
            td3 = TD3(env=reward_env,
                      max_action=reward_env.get_max_action(),
                      config=config)
            reward_list_train, _, _ = td3.train(reward_env, test_env=real_env)
            reward_list_test, _, _ = td3.test(real_env)
            avg_reward_test = statistics.mean(reward_list_test)

            unsolved_weight = config["agents"]["gtn"]["unsolved_weight"]
            score += len(reward_list_train) + max(0, (real_env.get_solved_reward()-avg_reward_test))*unsolved_weight

        score = score/NUM_EVALS

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
    run_id = 'pendulum_td3_params_bohb_' + x.strftime("%Y-%m-%d-%H") + '_2_after'

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
