import yaml
import torch
import sys
from agents.GTN import GTN_Worker
from automl.bohb_optim import run_bohb_parallel, run_bohb_serial
from communicate.tcp_worker_selector import start_communication_thread
import argparse


def my_parse():  # --bohb_id AAA --id BBB --moab_id CCC --port DDD --min_workers EEE --number_workers FFF --mode DDD
    parser = argparse.ArgumentParser()
    parser.add_argument("--bohb_id", type=int, default=50000)
    parser.add_argument("--id", type=int, default=1)
    parser.add_argument("--moab_id", type=str)
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--min_workers", type=int, default=2, help="Minimum number of workers that have to be active, before we otherwise abort")
    parser.add_argument("--number_workers", type=int, default=10000, help="Total number of workers for this experiment")
    parser.add_argument("--mode", type=str, choices=["worker", "master"], default=["worker"], help="either it is a worker or the one master")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = my_parse()
    print(vars(args))

    bohb_id = args.bohb_id
    id = args.id
    start_communication_thread(args=args)
    worker = GTN_Worker(bohb_id=bohb_id, id=id)
    worker.run()
