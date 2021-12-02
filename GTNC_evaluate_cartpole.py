import datetime
import time
import traceback
import yaml
import ConfigSpace as CS
import random
import numpy as np
import torch
from copy import deepcopy
from agents.GTN import GTN_Master
from automl.bohb_optim import run_bohb_parallel
from communicate.tcp_master_selector import start_communication_thread
import argparse
import os
from experiment_helpers.exp_logging import set_logger_up

import logging

logger = logging.getLogger()


def my_parse():  # --bohb_id AAA --id BBB --moab_id CCC --port DDD --min_workers EEE --number_workers FFF --mode DDD
    parser = argparse.ArgumentParser()
    parser.add_argument("--bohb_id", type=int, default=50000)
    parser.add_argument("--id", type=int, default=1)
    parser.add_argument("--moab_id", type=str)
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--min_workers", type=int, default=2, help="Minimum number of workers that have to be active, before we otherwise abort")
    parser.add_argument("--number_workers", type=int, default=10000, help="Total number of workers for this experiment")
    parser.add_argument("--mode", type=str, choices=["worker", "master"], default="worker", help="either it is a worker or the one master")
    args = parser.parse_args()
    return args


def seed_experiment(number):
    random.seed(number + int(time.time()))
    np.random.seed(number + int(time.time()))
    torch.manual_seed(number + int(time.time()))
    torch.cuda.manual_seed_all(number + int(time.time()))


class ExperimentWrapper():

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
        return config

    def compute(self, working_dir, bohb_id):
        with open("configurations/default_config_cartpole_syn_env.yaml", 'r') as stream:
            default_config = yaml.safe_load(stream)

        config = self.get_specific_config(default_config)

        try:
            gtn = GTN_Master(config, bohb_id=bohb_id, bohb_working_dir=working_dir)
            _, score_list = gtn.run()
            score = len(score_list)
            error = ""
        except:
            score = float('Inf')
            score_list = []
            error = traceback.format_exc()
            logger.info(error)

        info = {}
        info['error'] = str(error)
        info['score_list'] = str(score_list)

        logger.info('----------------------------')
        logger.info('FINAL SCORE: ' + str(score))
        logger.info('SCORE LIST:  ' + str(score_list))
        logger.info("END BOHB ITERATION")
        logger.info('----------------------------')

        return {
            "loss": score,
            "info": info
        }


def get_working_dir(run_id):
    return str(os.path.join(os.getcwd(), "results", run_id))


if __name__ == "__main__":
    args = my_parse()

    x = datetime.datetime.now()
    run_id = 'GTNC_evaluate_cartpole_' + x.strftime("%Y-%m-%d-%H")

    # LOGGING:
    set_logger_up(logger=logger, name=f"log_MASTER_id_{args.bohb_id}")
    logger.info(f"Starting: {run_id}")

    # COMMUNICATION:
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
    exp = ExperimentWrapper()

    w_dir = get_working_dir(run_id)

    exp.compute(working_dir=w_dir, bohb_id=args.bohb_id)
