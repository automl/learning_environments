import datetime
import sys
import yaml
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from copy import deepcopy
from agents.QL import QL
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

        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='ql_alpha', lower=0.01, upper=1, log=True, default_value=0.5))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='ql_gamma', lower=0.005, upper=0.5, log=True, default_value=0.1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='ql_eps_init', lower=0.01, upper=1, log=True, default_value=0.5))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='ql_eps_min', lower=0.005, upper=0.05, log=True, default_value=0.01))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='ql_eps_decay', lower=0.005, upper=0.5, log=True, default_value=0.01))

        return cs


    def get_specific_config(self, cso, default_config, budget):
        config = deepcopy(default_config)

        config["agents"]['ql']['alpha'] = cso["ql_alpha"]
        config["agents"]['ql']['gamma'] = 1-cso["ql_gamma"]
        config["agents"]['ql']['eps_init'] = cso["ql_eps_init"]
        config["agents"]['ql']['eps_min'] = cso["ql_eps_min"]
        config["agents"]['ql']['eps_decay'] = 1-cso["ql_eps_decay"]
        config["device"] = 'cuda'

        return config


    def compute(self, working_dir, bohb_id, config_id, cso, budget, *args, **kwargs):
        with open("default_config_gridworld.yaml", 'r') as stream:
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

        score = 0
        for i in range(NUM_EVALS):
            ql = QL(env=real_env, config=config)
            rewards, _ = ql.train(real_env)
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
    run_id = 'holeroomlarge_ql_params_bohb_' + x.strftime("%Y-%m-%d-%H")

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
