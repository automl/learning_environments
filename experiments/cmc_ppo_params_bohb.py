import datetime
import sys
import yaml
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from copy import deepcopy
from agents.PPO import PPO
from envs.env_factory import EnvFactory
from automl.bohb_optim import run_bohb_parallel, run_bohb_serial

NUM_EVALS = 3

class ExperimentWrapper():
    def get_bohb_parameters(self):
        params = {}
        params['min_budget'] = 1000
        params['max_budget'] = 8000
        params['eta'] = 2
        params['iterations'] = 1000
        params['random_fraction'] = 0.3

        return params


    def get_configspace(self):
        cs = CS.ConfigurationSpace()

        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='ppo_update_episodes', lower=1, upper=100, log=True, default_value=20))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='ppo_ppo_epochs', lower=20, upper=200, log=True, default_value=100))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='ppo_gamma', lower=0.001, upper=0.1, log=True, default_value=0.01))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='ppo_lr', lower=1e-4, upper=1e-2, log=True, default_value=3e-4))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='ppo_vf_coef', lower=0.1, upper=2, log=True, default_value=0.5))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='ppo_ent_coef', lower=0.002, upper=0.05, log=True, default_value=0.01))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='ppo_eps_clip', lower=0.05, upper=1, log=True, default_value=0.2))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='ppo_same_action_num', lower=1, upper=3, log=False, default_value=1))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='ppo_activation_fn', choices=['relu', 'tanh', 'leakyrelu', 'prelu'], default_value='relu'))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='ppo_hidden_size', lower=48, upper=192, log=True, default_value=128))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='ppo_hidden_layer', lower=1, upper=2, log=False, default_value=2))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='ppo_action_std', lower=0.1, upper=2, log=True, default_value=0.5))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='ppo_early_out_virtual_diff', lower=1e-2, upper=1e-1, log=True, default_value=3e-2))

        return cs


    def get_specific_config(self, cso, default_config, budget):
        config = deepcopy(default_config)

        config["agents"]["ppo"]["update_episodes"] = cso["ppo_update_episodes"]
        config["agents"]["ppo"]["ppo_epochs"] = cso["ppo_ppo_epochs"]
        config["agents"]["ppo"]["gamma"] = 1-cso["ppo_gamma"]
        config["agents"]["ppo"]["lr"] = cso["ppo_lr"]
        config["agents"]["ppo"]["vf_coef"] = cso["ppo_vf_coef"]
        config["agents"]["ppo"]["ent_coef"] = cso["ppo_ent_coef"]
        config["agents"]["ppo"]["eps_clip"] = cso["ppo_eps_clip"]
        config["agents"]["ppo"]["same_action_num"] = cso["ppo_same_action_num"]
        config["agents"]["ppo"]["activation_fn"] = cso["ppo_activation_fn"]
        config["agents"]["ppo"]["hidden_size"] = cso["ppo_hidden_size"]
        config["agents"]["ppo"]["hidden_layer"] = cso["ppo_hidden_layer"]
        config["agents"]["ppo"]["action_std"] = cso["ppo_action_std"]
        config["agents"]["ppo"]["early_out_virtual_diff"] = cso["ppo_early_out_virtual_diff"]
        config["device"] = 'cuda'

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

        config["agents"]["ppo"]["train_episodes"] = int(budget)

        info = {}

        # generate environment
        env_fac = EnvFactory(config)
        real_env = env_fac.generate_real_env()

        score = 0
        for i in range(NUM_EVALS):
            ppo = PPO(env=real_env,
                      config=config)
            rewards, _, _ = ppo.train(real_env)
            score += len(rewards)

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
    run_id = 'cmc_ppo_params_bohb_' + x.strftime("%Y-%m-%d-%H")

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
