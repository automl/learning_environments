import yaml
import torch
import sys
from agents.GTN import GTN_Worker
from automl.bohb_optim import run_bohb_parallel, run_bohb_serial

if __name__ == "__main__":
    torch.set_num_threads(1)

    for arg in sys.argv[1:]:
        print(arg)

    bohb_id = int(sys.argv[1])
    id = int(sys.argv[2])
    worker = GTN_Worker(bohb_id=bohb_id, id=id)
    worker.run()

