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

        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='gtn_noise_std', lower=1e-2, upper=1, log=True, default_value=1e-1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='gtn_step_size', lower=5e-1, upper=2, log=True, default_value=1))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='gtn_num_grad_evals', choices=[1,2,3], default_value=1))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='gtn_grad_eval_type', choices=['mean', 'minmax'], default_value='minmax'))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='gtn_weight_decay', lower=1e-2, upper=0.1, log=True, default_value=1e-1))

        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='gtn_score_transform_type', lower=4, upper=6, log=False, default_value=4))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='gtn_exploration_gain', choices=[0, 0.001], default_value=0))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='gtn_correct_path_gain', choices=[0, 0.001], default_value=0))

        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='ql_action_noise', lower=1e-2, upper=10, log=True, default_value=1e-1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='ql_action_noise_decay', lower=0.7, upper=1, log=True, default_value=0.9))


        #cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='env_hidden_size', choices=[32], default_value=32))
        #cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='env_hidden_layer', choices=[1], default_value=1))

        #cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='env_activation_fn', choices=['relu', 'tanh', 'leakyrelu', 'prelu'], default_value='relu'))

        return cs


    def get_specific_config(self, cso, default_config, budget):
        config = deepcopy(default_config)
        env = config["env_name"]

        config["render_env"] = False

        config["agents"]['gtn']['noise_std'] = cso["gtn_noise_std"]
        config["agents"]['gtn']['step_size'] = cso["gtn_step_size"]
        config["agents"]['gtn']['num_grad_evals'] = int(cso["gtn_num_grad_evals"])
        config["agents"]['gtn']['grad_eval_type'] = cso["gtn_grad_eval_type"]
        config["agents"]['gtn']['weight_decay'] = float(cso["gtn_weight_decay"])
        config["agents"]['gtn']['score_transform_type'] = int(cso["gtn_score_transform_type"])
        config["agents"]['gtn']['exploration_gain'] = float(cso["gtn_exploration_gain"])
        config["agents"]['gtn']['correct_path_gain'] = float(cso["gtn_correct_path_gain"])

        config["agents"]['ql']['action_noise'] = float(cso["ql_action_noise"])
        config["agents"]['ql']['action_noise_decay'] = float(cso["ql_action_noise_decay"])

        #config["envs"][env]['hidden_size'] = int(cso["env_hidden_size"])
        #config["envs"][env]['hidden_layer'] = int(cso["env_hidden_layer"])



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

        try:
            gtn = GTN_Master(config, bohb_id=bohb_id)
            _, score_list = gtn.run()
            score = len(score_list)
            error = ""
        except:
            score = float('Inf')
            score_list = []
            error = traceback.format_exc()
            print(error)

        info = {}
        info['error'] = str(error)
        info['score_list'] = str(score_list)

        print('----------------------------')
        print('FINAL SCORE: ' + str(score))
        print('SCORE LIST:  ' + str(score_list))
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
    run_id = 'GTNC_params_bohb_' + x.strftime("%Y-%m-%d-%H")

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
