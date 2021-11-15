import datetime
import os
import time
import sys
import traceback
import yaml
import ConfigSpace as CS
import random
import numpy as np
import torch
from copy import deepcopy
from agents.GTN import GTN_Master
import argparse
from communicate.tcp_master_selector import start_communication_thread


class ExperimentWrapper():
    def __init__(self, reward_env_type):
        self.reward_env_type = reward_env_type

    def get_bohb_parameters(self):
        params = {}
        params['min_budget'] = 1
        params['max_budget'] = 1
        params['eta'] = 2
        params['random_fraction'] = 1
        params['iterations'] = 10000

        return params

    def get_configspace(self):
        cs = CS.ConfigurationSpace()
        return cs

    def get_specific_config(self, default_config):
        config = deepcopy(default_config)
        global reward_env_type
        config["envs"]['HalfCheetah-v3']['reward_env_type'] = self.reward_env_type
        config["envs"]['HalfCheetah-v3']['solved_reward'] = sys.maxsize  # AUC
        # config["envs"]['HalfCheetah-v3']['device'] = "cuda:0"  # AUC
        return config

    def compute(self, working_dir, bohb_id):
        with open("configurations/default_config_halfcheetah_td3_se_opt.yaml", 'r') as stream:
            default_config = yaml.safe_load(stream)

        config = self.get_specific_config(default_config)

        print('----------------------------')
        print("START BOHB ITERATION")
        print('CONFIG: ' + str(config))
        print('----------------------------')

        try:
            gtn = GTN_Master(config, bohb_id=bohb_id, bohb_working_dir=working_dir)
            _, score_list, model_name = gtn.run()
            score = -sorted(score_list)[-1]
            error = ""
        except:
            score = float('-Inf')
            score_list = []
            model_name = None
            error = traceback.format_exc()
            print(error)

        info = {}
        info['error'] = str(error)
        info['config'] = str(config)
        info['score_list'] = str(score_list)
        info['model_name'] = str(model_name)

        print('----------------------------')
        print('FINAL SCORE: ' + str(score))
        print('SCORE LIST:  ' + str(score_list))
        print("END BOHB ITERATION")
        print('----------------------------')

        return {
            "loss": score,
            "info": info
        }


def my_parse():  # --bohb_id AAA --id BBB --moab_id CCC --port DDD --min_workers EEE --number_workers FFF --mode DDD
    parser = argparse.ArgumentParser()
    parser.add_argument("--bohb_id", type=int, default=50000)
    parser.add_argument("--id", type=int, default=1)
    parser.add_argument("--moab_id", type=str)
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--min_workers", type=int, default=2, help="Minimum number of workers that have to be active, before we otherwise abort")
    parser.add_argument("--number_workers", type=int, default=10000, help="Total number of workers for this experiment")
    parser.add_argument("--mode", type=str, choices=["worker", "master"], default="worker", help="either it is a worker or the one master")
    parser.add_argument("--reward_env_type", type=int, default=5)
    args = parser.parse_args()
    return args


def seed_experiment(number):
    random.seed(number + int(time.time()))
    np.random.seed(number + int(time.time()))
    torch.manual_seed(number + int(time.time()))
    torch.cuda.manual_seed_all(number + int(time.time()))


def get_working_dir(run_id):
    return str(os.path.join(os.getcwd(), "results", run_id))


if __name__ == "__main__":
    x = datetime.datetime.now()
    args = my_parse()
    id = args.id

    run_id = 'syn_env_learn_halfcheetah_' + x.strftime("%Y-%m-%d-%H")

    args = my_parse()

    start_communication_thread(args=args)

    seed_experiment(number=args.bohb_id)

    # This is the call with BOHB:
    """
    res = run_bohb_parallel(id=args.bohb_id,
                            bohb_workers=1,  # this should be set to 1, otherwise the program does not execute
                            run_id=run_id,
                            experiment_wrapper=ExperimentWrapper())    
    """

    # This is the call without any BOHB:
    exp = ExperimentWrapper(reward_env_type=args.reward_env_type)

    w_dir = get_working_dir(run_id)

    exp.compute(working_dir=w_dir, bohb_id=args.bohb_id)
