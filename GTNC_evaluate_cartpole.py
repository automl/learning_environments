import datetime
import time
import traceback
import yaml
import random
import numpy as np
import torch
from copy import deepcopy
from agents.GTN import GTN_Master
from communicate.tcp_master_selector import start_communication_thread
import argparse
from automl.bohb_optim import get_working_dir


def my_parse():  # --bohb_id AAA --id BBB --moab_id CCC --port DDD --min_workers EEE --number_workers FFF --mode DDD
    parser = argparse.ArgumentParser()
    parser.add_argument("--bohb_id", type=int, default=5000)
    parser.add_argument("--id", type=int)
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


def compute(self, working_dir, bohb_id):
    with open("default_config_cartpole_syn_env.yaml", 'r') as stream:
        default_config = yaml.safe_load(stream)

    config = deepcopy(default_config)

    try:
        gtn = GTN_Master(config, bohb_id=bohb_id, bohb_working_dir=working_dir)
        _, score_list = gtn.run()
        score = len(score_list)
        error = ""
    except:
        score = float('Inf')
        score_list = []
        error = traceback.format_exc()
        print(error)

    info = {'error': str(error), 'score_list': str(score_list)}

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
    x = datetime.datetime.now()
    run_id = 'GTNC_evaluate_cartpole_' + x.strftime("%Y-%m-%d-%H")

    args = my_parse()

    start_communication_thread(args=args)

    seed_experiment(number=args.bohb_id)

    # get BOHB log directory
    working_dir = get_working_dir(run_id)

    res = compute(working_dir=working_dir, bohb_id=args.bohb_id)
    print("FINAL: ", res)
