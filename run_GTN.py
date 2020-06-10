import numpy as np
import torch
import yaml

from agents.GTN import GTN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    with open("default_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    # set seeds
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    reptile = GTN(config)
    reptile.run()
