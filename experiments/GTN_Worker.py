import yaml
import torch
import sys
from agents.GTN import GTN_Worker
from automl.bohb_optim import run_bohb_parallel, run_bohb_serial

if __name__ == "__main__":
    with open("../default_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    gtn_config = config['agents']['gtn']
    torch.set_num_threads(gtn_config['num_threads_per_worker'])

    for arg in sys.argv[1:]:
        print(arg)

    bohb_id = int(sys.argv[1])
    id = int(sys.argv[2])
    worker = GTN_Worker(config=config, bohb_id=bohb_id, id=id)

