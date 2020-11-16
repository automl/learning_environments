import yaml
import torch
import multiprocessing as mp
import sys
import time
from agents.GTN_master import GTN_Master
from agents.GTN_worker import GTN_Worker


def run_gtn_on_single_pc(config):
    def run_gtn_worker(id):
        gtn = GTN_Worker(id)
        gtn.run()

    def run_gtn_master(config):
        gtn = GTN_Master(config)
        gtn.run()

    p_list = []

    # cleanup working directory from old files
    gtn_base = GTN_Master(config)
    gtn_base.clean_working_dir()
    time.sleep(2)

    # first start master
    p = mp.Process(target=run_gtn_master, args=(config,))
    p.start()
    p_list.append(p)

    # then start workers
    num_workers = config["agents"]["gtn"]["num_workers"]
    for id in range(num_workers):
        p = mp.Process(target=run_gtn_worker, args=(id,))
        p.start()
        p_list.append(p)

    # wait till everything has finished
    for p in p_list:
        p.join()


def run_gtn_on_multiple_pcs(config, id):
    if id == -1:
        gtn_master = GTN_Master(config)
        gtn_master.clean_working_dir()
        gtn_master.run()
    elif id >= 0:
        gtn_worker = GTN_Worker(id)
        gtn_worker.run()
    else:
        raise ValueError("Invalid ID")


if __name__ == "__main__":
    with open("../default_config_gridworld.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    gtn_config = config['agents']['gtn']
    mode = 'single'

    torch.set_num_threads(gtn_config['num_threads_per_worker'])

    if mode == 'single':
        run_gtn_on_single_pc(config)
    elif mode == 'multi':
        for arg in sys.argv[1:]:
            print(arg)

        id = int(sys.argv[1])
        run_gtn_on_multiple_pcs(config, id)

