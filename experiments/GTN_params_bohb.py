import datetime
import sys
import traceback
import yaml
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from copy import deepcopy
from agents.GTN import GTN_Master, GTN_Worker
from automl.bohb_optim import run_bohb_parallel, run_bohb_serial


class ExperimentWrapper():
    def get_bohb_parameters(self):
        params = {}
        params['seed'] = 42
        params['min_budget'] = 1
        params['max_budget'] = 2
        params['eta'] = 2
        params['random_fraction'] = 0.3
        params['iterations'] = 1000

        return params


    def get_configspace(self):
        cs = CS.ConfigurationSpace()

        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='gtn_max_iterations', choices=[10,30], default_value=10))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='gtn_noise_std', lower=1e-3, upper=1e-1, log=True, default_value=1e-2))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='gtn_learning_rate', lower=1e-2, upper=1, log=True, default_value=1e-1))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='gtn_virtual_env_reps', choices=[1,3,10], default_value=1))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='env_activation_fn', choices=['relu', 'tanh', 'leakyrelu', 'prelu'], default_value='relu'))

        return cs


    def get_specific_config(self, cso, default_config, budget):
        config = deepcopy(default_config)

        config["render_env"] = False

        config["agents"]['gtn']['noise_std'] = cso["gtn_noise_std"]
        config["agents"]['gtn']['learning_rate'] = cso["gtn_learning_rate"]
        config["agents"]['gtn']['virtual_env_reps'] = cso["virtual_env_reps"]
        config["envs"]['Bandit-v0']['activation_fn'] = cso["env_activation_fn"]

        return config


    def compute(self, working_dir, bohb_id, config_id, cso, budget, *args, **kwargs):
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
        score = None
        error = ""
        try:
            gtn = GTN_Master(config, bohb_id=bohb_id)
            score = gtn.run()
        except:
            score = float('Inf')
            error = traceback.format_exc()
            print(error)

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
    run_id = 'GTN_params_bohb_' + x.strftime("%Y-%m-%d-%H")

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            print(arg)
        res = run_bohb_parallel(id=int(sys.argv[1]),
                                bohb_workers=int(sys.argv[2]),
                                run_id=run_id,
                                experiment_wrapper=ExperimentWrapper())
    else:
        res = run_bohb_serial(run_id=run_id,
                              experiment_wrapper=ExperimentWrapper())
