import torch
import sys
import os
from agents.GTN import GTN_Worker
import multiprocessing as mp

script_dir = os.path.dirname(os.path.abspath(__file__))
par_dir = os.path.join(script_dir, os.pardir)
sys.path.append(par_dir)
os.chdir(par_dir)

if __name__ == "__main__":
    torch.set_num_threads(1)

    gtn = GTN_Worker(bohb_id=0, id=0)
    gtn.clean_working_dir()


    def run_gtn_worker(id):
        gtn = GTN_Worker(bohb_id=0, id=id)
        gtn.run()


    num_workers = 16
    p_list = []

    for id in range(num_workers):
        p = mp.Process(target=run_gtn_worker, args=(id,))
        p.start()
        p_list.append(p)
