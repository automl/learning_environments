import torch
import sys
from agents.GTN import GTN_Worker
import multiprocessing as mp

if __name__ == "__main__":
    torch.set_num_threads(1)

    def run_gtn_worker(id):
        gtn = GTN_Worker(id)
        gtn.run()

    num_workers = 16
    p_list = []

    for id in range(num_workers):
        p = mp.Process(target=run_gtn_worker, args=(id,))
        p.start()
        p_list.append(p)

    for arg in sys.argv[1:]:
        print(arg)

    bohb_id = int(sys.argv[1])
    id = int(sys.argv[2])
    worker = GTN_Worker(bohb_id=bohb_id, id=id)
    worker.run()

