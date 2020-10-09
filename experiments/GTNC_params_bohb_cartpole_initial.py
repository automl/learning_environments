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

        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='gtn_noise_std', lower=1e-3, upper=1, log=True, default_value=1e-2))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='gtn_step_size', lower=1e-1, upper=2, log=True, default_value=1))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='gtn_mirrored_sampling', choices=[False, True], default_value=True))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='gtn_num_grad_evals', choices=[1,2,3], default_value=1))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='gtn_grad_eval_type', choices=['mean', 'minmax'], default_value='minmax'))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='gtn_score_transform_type', lower=0, upper=6, log=False, default_value=5))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='gtn_weight_decay', lower=1e-4, upper=1e-1, log=False, default_value=1e-2))

        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='ddqn_test_episodes', lower=1, upper=50, log=True, default_value=10))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='ddqn_lr', lower=3e-4, upper=3e-3, log=True, default_value=1e-3))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='ddqn_gamma', lower=0.001, upper=0.1, log=True, default_value=0.01))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='ddqn_tau', lower=0.005, upper=0.05, log=True, default_value=0.01))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='ddqn_eps_init', lower=0.01, upper=1, log=True, default_value=0.9))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='ddqn_eps_min', lower=0.001, upper=0.1, log=True, default_value=0.01))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='ddqn_eps_decay', lower=0.001, upper=0.3, log=True, default_value=0.1))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='ddqn_batch_size', lower=64, upper=256, log=True, default_value=128))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='ddqn_hidden_size', lower=64, upper=256, log=True, default_value=128))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='ddqn_hidden_layer', lower=1, upper=2, log=False, default_value=2))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='ddqn_activation_fn', choices=['relu', 'tanh', 'leakyrelu', 'prelu'], default_value='relu'))

        return cs


    def get_specific_config(self, cso, default_config, budget):
        config = deepcopy(default_config)

        config["render_env"] = False

        config["agents"]['gtn']['noise_std'] = cso["gtn_noise_std"]
        config["agents"]['gtn']['step_size'] = cso["gtn_step_size"]
        config["agents"]['gtn']['mirrored_sampling'] = int(cso["gtn_mirrored_sampling"])
        config["agents"]['gtn']['num_grad_evals'] = int(cso["gtn_num_grad_evals"])
        config["agents"]['gtn']['grad_eval_type'] = cso["gtn_grad_eval_type"]
        config["agents"]['gtn']['score_transform_type'] = int(cso["gtn_score_transform_type"])
        config["agents"]['gtn']['weight_decay'] = float(cso["gtn_weight_decay"])

        config["agents"]['ddqn']['test_episodes'] = int(cso["ddqn_test_episodes"])
        config["agents"]['ddqn']['lr'] = float(cso["ddqn_lr"])
        config["agents"]['ddqn']['gamma'] = 1-float(cso["ddqn_gamma"])
        config["agents"]['ddqn']['tau'] = float(cso["ddqn_tau"])
        config["agents"]['ddqn']['eps_init'] = float(cso["ddqn_eps_init"])
        config["agents"]['ddqn']['eps_min'] = int(cso["ddqn_eps_min"])
        config["agents"]['ddqn']['eps_decay'] = 1-float(cso["ddqn_eps_decay"])
        config["agents"]['ddqn']['batch_size'] = int(cso["ddqn_batch_size"])
        config["agents"]['ddqn']['hidden_size'] = int(cso["ddqn_hidden_size"])
        config["agents"]['ddqn']['hidden_layer'] = int(cso["ddqn_hidden_layer"])
        config["agents"]['ddqn']['activation_fn'] = cso["ddqn_activation_fn"]

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
    run_id = 'GTNC_params_bohb_cartpole_initial_' + x.strftime("%Y-%m-%d-%H")

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
