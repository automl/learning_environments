import datetime
import sys
import traceback
import yaml
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from copy import deepcopy
from agents.REPTILE import REPTILE
from agents.agent_utils import test
from automl.bohb_optim import run_bohb_parallel, run_bohb_serial


class ExperimentWrapper():
    def get_bohb_parameters(self):
        params = {}
        params['seed'] = 42
        params['min_budget'] = 1
        params['max_budget'] = 2
        params['eta'] = 2
        params['iterations'] = 1000
        params['random_fraction'] = 1

        return params


    def get_configspace(self):
        cs = CS.ConfigurationSpace()

        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='rep_max_iterations', choices=[0, 1, 5, 10, 20]))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='rep_step_size', choices=[0.01, 0.1, 1]))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='rep_parallel_update', choices=[False, True]))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='rep_env_num', choices=[1,3,10]))

        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='ddqn_max_episodes', choices=[0,1,2,5,10,20,50,100]))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='ddqn_dropout', choices=[0, 0.05, 0.1, 0.2]))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='ddqn_activation_fn', choices=['relu', 'leakyrelu', 'tanh']))

        return cs


    def get_specific_config(self, cso, default_config, budget):
        config = deepcopy(default_config)

        config["render_env"] = False

        config["agents"]['reptile']['max_iterations'] = cso["rep_max_iterations"]
        config["agents"]['reptile']['step_size'] = cso["rep_step_size"]
        config["agents"]['reptile']['parallel_update'] = cso["rep_parallel_update"]
        config["agents"]['reptile']['env_num'] = cso["rep_env_num"]

        config["agents"]["ddqn"]["max_episodes"] = cso["ddqn_max_episodes"]
        config["agents"]["ddqn"]["dropout"] = cso["ddqn_dropout"]
        config["agents"]["ddqn"]["activation_fn"] = cso["ddqn_activation_fn"]

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
        episodes_till_solved = None
        error = ""

        try:
            reptile = REPTILE(config)
            reptile.train()
            score, episodes_till_solved = test(agent=reptile.agent,
                                               env_factory=reptile.env_factory,
                                               config=reptile.config,
                                               num_envs=10)
        except:
            score = float('Inf')
            error = traceback.format_exc()
            print(error)

        info['config'] = str(config)
        info['episodes_till_solved'] = str(episodes_till_solved)
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
    # SEED = 42
    # random.seed(SEED)
    # np.random.seed(SEED)
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed_all(SEED)

    x = datetime.datetime.now()
    run_id = 'REPTILE_params_bohb_' + x.strftime("%Y-%m-%d-%H")

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
