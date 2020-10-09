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

        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='gtn_noise_std', lower=1e-3, upper=1, log=True, default_value=1e-1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='gtn_step_size', lower=1e-1, upper=2, log=True, default_value=1))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='gtn_mirrored_sampling', choices=[False, True], default_value=True))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='gtn_num_grad_evals', choices=[1,2], default_value=1))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='gtn_grad_eval_type', choices=['mean', 'minmax'], default_value='minmax'))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='gtn_score_transform_type', lower=0, upper=6, log=False, default_value=5))

        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='td3_test_episodes', lower=1, upper=50, log=True, default_value=10))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_lr', lower=5e-4, upper=5e-2, log=False, default_value=1e-3))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_gamma', lower=0.001, upper=0.1, log=True, default_value=0.01))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_tau', lower=0.005, upper=0.05, log=False, default_value=0.01))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='td3_batch_size', lower=64, upper=256, log=True, default_value=128))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='td3_policy_delay', lower=2, upper=4, log=False, default_value=2))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='td3_hidden_size', lower=64, upper=256, log=True, default_value=128))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='td3_hidden_layer', lower=1, upper=2, log=False, default_value=2))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='td3_activation_fn', choices=['relu', 'tanh', 'leakyrelu', 'prelu'], default_value='relu'))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_action_std', lower=0.1, upper=0.4, log=True, default_value=0.2))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_policy_std', lower=0.1, upper=0.4, log=True, default_value=0.2))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_policy_std_clip', lower=0.1, upper=1, log=True, default_value=0.5))

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

        config["agents"]['td3']['test_episodes'] = float(cso["td3_test_episodes"])
        config["agents"]['td3']['lr'] = float(cso["td3_lr"])
        config["agents"]['td3']['gamma'] = 1-float(cso["td3_gamma"])
        config["agents"]['td3']['tau'] = float(cso["td3_tau"])
        config["agents"]['td3']['batch_size'] = float(cso["td3_batch_size"])
        config["agents"]['td3']['policy_delay'] = int(cso["td3_policy_delay"])
        config["agents"]['td3']['hidden_size'] = int(cso["td3_hidden_size"])
        config["agents"]['td3']['hidden_layer'] = int(cso["td3_hidden_layer"])
        config["agents"]['td3']['activation_fn'] = cso["td3_activation_fn"]
        config["agents"]['td3']['action_std'] = int(cso["td3_action_std"])
        config["agents"]['td3']['policy_std'] = int(cso["td3_policy_std"])
        config["agents"]['td3']['policy_std_clip'] = int(cso["td3_policy_std_clip"])

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
    run_id = 'GTNC_params_bohb_pendulum_initial_' + x.strftime("%Y-%m-%d-%H")

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
