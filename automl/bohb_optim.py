import os
import sys

script_dir = os.path.dirname(os.path.abspath( __file__ ))
par_dir = os.path.join(script_dir, os.pardir)

sys.path.append(par_dir)
os.chdir(par_dir)

import tensorflow as tf
import random
import numpy as np
import time
import yaml
import torch
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
import psutil
from hpbandster.core.worker import Worker
from hpbandster.core.master import Master
from hpbandster.optimizers.iterations import SuccessiveHalving
from hpbandster.optimizers.config_generators.bohb import BOHB as BOHB
from copy import deepcopy
from agents.GTN import GTN

SEED = 42

BOHB_MIN_BUDGET = 1
BOHB_MAX_BUDGET = 8
BOHB_ETA = 2
BOHB_WORKERS = 16
BOHB_ITERATIONS = 100000

def get_configspace():
    cs = CS.ConfigurationSpace()

    cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='gtn_match_lr', lower=1e-5, upper=1e-3, log=True, default_value=1e-4))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='gtn_match_weight_decay', lower=1e-9, upper=1e-6, log=True, default_value=1e-9))
    cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='gtn_match_iterations', lower=10, upper=1000, log=True, default_value=500))
    cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='gtn_real_iterations', lower=1, upper=10, log=True, default_value=1))
    cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='gtn_virtual_iterations', lower=1, upper=10, log=True, default_value=1))
    cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='gtn_different_envs', lower=1, upper=5, log=True, default_value=1))

    cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_lr', lower=1e-5, upper=1e-3, log=True, default_value=3e-4))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='td3_weight_decay', lower=1e-9, upper=1e-6, log=True, default_value=1e-9))
    cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='td3_optim_env_with_ac', lower=0, upper=2, log=False, default_value=0))
    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='tde_activation_fn', choices=['relu','tanh'], default_value='relu'))

    return cs

def construct_specific_config(cso, default_config, budget):
    config = deepcopy(default_config)

    config["agents"]["gtn"]["max_iterations"]     = int(budget)
    config["agents"]["gtn"]["match_iterations"]   = cso["gtn_match_iterations"]
    config["agents"]["gtn"]["real_iterations"]    = cso["gtn_real_iterations"]
    config["agents"]["gtn"]["virtual_iterations"] = cso["gtn_virtual_iterations"]
    config["agents"]["gtn"]["different_envs"]     = cso["gtn_different_envs"]

    config["agents"]["td3"]["lr"]                 = cso["td3_lr"]
    config["agents"]["td3"]["weight_decay"]       = cso["td3_weight_decay"]
    config["agents"]["td3"]["optim_env_with_ac"]  = cso["td3_optim_env_with_ac"]
    config["agents"]["td3"]["activation_fn"]      = cso["tde_activation_fn"]


    return config

class BOHBWorker(Worker):
    def __init__(self, working_dir, *args, **kwargs):
        super(BOHBWorker, self).__init__(*args, **kwargs)
        self.session = tf.Session()
        print(kwargs)
        self.working_dir = working_dir

        print(os.getcwd())
        with open("default_config.yaml", 'r') as stream:
            self.default_config = yaml.safe_load(stream)

    def compute(self, config_id, config, budget, *args, **kwargs):
        config = construct_specific_config(config, self.default_config, budget)
        print('----------------------------')
        print("START BOHB ITERATION")
        print('CONFIG: ' + str(config))
        print('BUDGET: ' + str(budget))
        print('MODEL CONFIG: ' + str(config))
        print('----------------------------')

        info = {}
        gtn = GTN(config)
        gtn.run()
        score = gtn.validate()

        info['config'] = str(config)

        print('----------------------------')
        print('FINAL SCORE: ' + str(score))
        print("END BOHB ITERATION")
        print('----------------------------')


        return {
            "loss": -score,
            "info": info
        }

class BohbWrapper(Master):
    def __init__(self, configspace=None,
                 eta=3, min_budget=0.01, max_budget=1,
                 min_points_in_model=None, top_n_percent=15,
                 num_samples=64, random_fraction=1 / 3, bandwidth_factor=3,
                 min_bandwidth=1e-3,
                 **kwargs):
        # TODO: Proper check for ConfigSpace object!
        if configspace is None:
            raise ValueError("You have to provide a valid CofigSpace object")

        cg = BOHB(configspace=configspace,
                  min_points_in_model=min_points_in_model,
                  top_n_percent=top_n_percent,
                  num_samples=num_samples,
                  random_fraction=random_fraction,
                  bandwidth_factor=bandwidth_factor,
                  min_bandwidth=min_bandwidth
                  )

        super().__init__(config_generator=cg, **kwargs)

        # Hyperband related stuff
        self.eta = eta
        self.min_budget = min_budget
        self.max_budget = max_budget

        # precompute some HB stuff
        self.max_SH_iter = -int(np.log(min_budget / max_budget) / np.log(eta)) + 1
        self.budgets = max_budget * np.power(eta, -np.linspace(self.max_SH_iter - 1, 0,
                                                               self.max_SH_iter))

        self.config.update({
            'eta': eta,
            'min_budget': min_budget,
            'max_budget': max_budget,
            'budgets': self.budgets,
            'max_SH_iter': self.max_SH_iter,
            'min_points_in_model': min_points_in_model,
            'top_n_percent': top_n_percent,
            'num_samples': num_samples,
            'random_fraction': random_fraction,
            'bandwidth_factor': bandwidth_factor,
            'min_bandwidth': min_bandwidth
        })

    def get_next_iteration(self, iteration, iteration_kwargs={}):
        # number of 'SH rungs'
        s = self.max_SH_iter - 1
        # number of configurations in that bracket
        n0 = int(np.floor((self.max_SH_iter) / (s + 1)) * self.eta ** s)
        ns = [max(int(n0 * (self.eta ** (-i))), 1) for i in range(s + 1)]

        return (SuccessiveHalving(HPB_iter=iteration, num_configs=ns,
                                  budgets=self.budgets[(-s - 1):],
                                  config_sampler=self.config_generator.get_config,
                                  **iteration_kwargs))


def get_bohb_interface():
    addrs = psutil.net_if_addrs()
    if 'eth0' in addrs.keys():
        print('FOUND eth0 INTERFACE')
        return 'eth0'
    else:
        print('FOUND lo INTERFACE')
        return 'lo'


def get_working_dir(run_id):
    return str(os.path.join(os.getcwd(), "experiments", run_id))


def runBohbParallel(id, run_id):
    # get suitable interface (eth0 or lo)
    bohb_interface = get_bohb_interface()

    # get BOHB log directory
    working_dir = get_working_dir(run_id)

    # every process has to lookup the hostname
    host = hpns.nic_name_to_host(bohb_interface)

    os.makedirs(working_dir, exist_ok=True)

    if int(id) > 0:
        print('START NEW WORKER')
        time.sleep(10)
        w = BOHBWorker(host=host,
                       run_id=run_id,
                       working_dir=working_dir)
        w.load_nameserver_credentials(working_directory=working_dir)
        w.run(background=False)
        exit(0)

    print('START NEW MASTER')
    ns = hpns.NameServer(run_id=run_id,
                         host=host,
                         port=0,
                         working_directory=working_dir)
    ns_host, ns_port = ns.start()

    w = BOHBWorker(host=host,
                   nameserver=ns_host,
                   nameserver_port=ns_port,
                   run_id=run_id,
                   working_dir=working_dir)
    w.run(background=True)

    result_logger = hpres.json_result_logger(directory=working_dir,
                                             overwrite=True)

    bohb = BohbWrapper(
        configspace=get_configspace(),
        run_id=run_id,
        eta=BOHB_ETA,
        host=host,
        nameserver=ns_host,
        nameserver_port=ns_port,
        min_budget=BOHB_MIN_BUDGET,
        max_budget=BOHB_MAX_BUDGET,
        result_logger=result_logger)

    res = bohb.run(n_iterations=BOHB_ITERATIONS,
                   min_n_workers=BOHB_WORKERS)
#    res = bohb.run(n_iterations=BOHB_ITERATIONS)

    bohb.shutdown(shutdown_workers=True)
    ns.shutdown()

    return res


def runBohbSerial(run_id):
    # get BOHB log directory
    working_dir = get_working_dir(run_id)

    # assign random port in the 30000-40000 range to avoid using a blocked port because of a previous improper bohb shutdown
    port = int(30000 + random.random() * 10000)

    ns = hpns.NameServer(run_id=run_id, host="127.0.0.1", port=port)
    ns.start()

    w = BOHBWorker(nameserver="127.0.0.1",
                   run_id=run_id,
                   nameserver_port=port,
                   working_dir=working_dir)
    w.run(background=True)

    result_logger = hpres.json_result_logger(directory=working_dir,
                                             overwrite=True)

    bohb = BohbWrapper(
        configspace=get_configspace(),
        run_id=run_id,
        eta=BOHB_ETA,
        min_budget=BOHB_MIN_BUDGET,
        max_budget=BOHB_MIN_BUDGET,
        nameserver="127.0.0.1",
        nameserver_port=port,
        result_logger=result_logger)

    res = bohb.run(n_iterations=BOHB_ITERATIONS)
    bohb.shutdown(shutdown_workers=True)
    ns.shutdown()

    return res


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    tf.set_random_seed(SEED)

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            print(arg)
        res = runBohbParallel(id=sys.argv[1], run_id=sys.argv[2])
    else:
        res = runBohbSerial(run_id='GTN')


