import argparse
import yaml
import torch
import torch.nn as nn
import multiprocessing as mp
import os
import time
import datetime
from agents.agent_utils import select_agent, test
from envs.env_factory import EnvFactory

parser = argparse.ArgumentParser(description='GTN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p', '--purge', default=False, type=bool, help='purge workspace')
parser.add_argument('-i', '--id', default=-1, type=int, help='id of the population member')
args = parser.parse_args()


class GTN(nn.Module):
    def __init__(self, config, id=0):
        super().__init__()

        # for saving/loading
        self.config = config

        gtn_config = config["agents"]["gtn"]
        self.max_iterations = gtn_config["max_iterations"]
        self.population_size = gtn_config["population_size"]
        self.noise_std = gtn_config["noise_std"]
        self.learning_rate = gtn_config["learning_rate"]
        self.time_mult = gtn_config["time_mult"]
        self.time_max = gtn_config["time_max"]
        self.agent_name = gtn_config["agent_name"]
        self.device = config["device"]

        self.env_factory = EnvFactory(config)
        self.virtual_env_orig = self.env_factory.generate_virtual_env()
        self.virtual_env = self.env_factory.generate_virtual_env()
        self.eps = self.env_factory.generate_virtual_env()
        self.noise = self.env_factory.generate_virtual_env()

        # for identifying the different members of a population
        self.id = id

        x = datetime.datetime.now()
        self.working_dir = str(os.path.join(os.getcwd(), "results", 'GTN_' + x.strftime("%Y-%m-%d-%H")))
        os.makedirs(self.working_dir, exist_ok=True)

        # for early out
        self.avg_runtime = None

    def get_working_dir(self):
        return str(os.path.join(os.getcwd(), "results", "GTN"))

    def calc_random_eps_and_noise(self):
        for l_virt, l_eps, l_noise in zip(self.virtual_env.modules(), self.eps.modules(), self.noise.modules()):
            if isinstance(l_virt, nn.Linear):
                l_eps.weight = torch.nn.Parameter(torch.normal(mean=torch.zeros_like(l_virt.weight),
                                                               std=torch.ones_like(l_virt.weight)))
                l_noise.weight = torch.nn.Parameter(l_eps.weight * self.noise_std)

    def add_noise_to_virtual_env(self, add=True):
        for l_orig, l_virt, l_noise in zip(self.virtual_env_orig.modules(), self.virtual_env.modules(), self.noise.modules()):
            if isinstance(l_virt, nn.Linear):
                if add:
                    l_virt.weight = torch.nn.Parameter(l_orig.weight + l_noise.weight)
                else:
                    l_virt.weight = torch.nn.Parameter(l_orig.weight - l_noise.weight)

    def subtract_noise_from_virtual_env(self):
        self.add_noise_to_virtual_env(add=False)

    def test_agent_on_real_env(self, agent, time_start, timeout):
        mean_episodes_till_solved, episodes_till_solved = test(agent=agent,
                                                               env_factory=self.env_factory,
                                                               config=self.config,
                                                               time_start=time_start,
                                                               timeout=timeout)
        return mean_episodes_till_solved

    def write_result_to_file(self, loss, time_elapsed, it):
        file_name = os.path.join(self.working_dir, str(it) + "_" + str(self.id))
        save_dict = {}
        save_dict["virtual_env"] = self.virtual_env.state_dict()
        save_dict["time_elapsed"] = time_elapsed
        save_dict["loss"] = loss
        torch.save(save_dict, file_name)

    def purge(self):
        pass


    def run(self):
        with torch.no_grad():
            for it in range(self.max_iterations):
                self.calc(it)
                self.update()



    def calc(self, it):
        # calculate timeout
        if it == 0:
            timeout = self.time_max
        else:
            timeout = max(self.avg_runtime*self.time_mult, self.time_max)

        time_start = time.time()

        agent_add = select_agent(config, self.agent_name)
        agent_sub = select_agent(config, self.agent_name)
        agent_sub.set_state_dict(agent_add.set_state_dict())

        self.calc_random_eps_and_noise()

        # first mirrored noise +N
        self.add_noise_to_virtual_env()
        agent_add.train(self.virtual_env, time_start=time_start, timeout=timeout)
        loss_add = self.test_agent_on_real_env(agent_add, time_start=time_start, timeout=timeout)

        # second mirrored noise +N
        self.subtract_noise_from_virtual_env()
        agent_sub.train(env=self.virtual_env, time_start=time_start, timeout=timeout)
        loss_sub = self.test_agent_on_real_env(agent_sub, time_start=time_start, timeout=timeout)

        if loss_add < loss_sub:
            self.subtract_noise_from_virtual_env()

        self.write_result_to_file(loss = min(loss_add, loss_sub), time_elapsed = time.time()-time.start, it=it)

    def update(self):

        # read agent results from file
        # update own weights



    def save(self, path):
        # not sure if working
        state = {}
        state["config"] = self.config
        state["agent"] = self.agent.get_state_dict()
        state["virtual_env"] = self.virtual_env.get_state_dict()
        state["input_seeds"] = self.input_seeds
        state["real_envs"] = []
        for real_env in self.real_envs:
            state["real_envs"].append(real_env.get_state_dict())
        torch.save(state, path)

    def load(self, path):
        # not sure if working
        if os.path.isfile(path):
            state = torch.load(self.path)
            self.__init__(state["config"])
            self.agent.set_state_dict(state["agent"])
            self.virtual_env.set_state_dict(state["virtual_env"])
            self.input_seeds = state["input_seeds"]
            for i in range(len(self.real_envs)):
                self.real_envs[i].set_state_dict(state["real_envs"][i])
        else:
            raise FileNotFoundError("File not found: " + str(path))


def run_gtn_on_single_pc(config):
    '''
    does the same as extract_download_links(), but in parallel
    '''

    def run_gtn(config, id):
        gtn = GTN(config, id)
        gtn.run()

    p_list = []
    population_size = config["gtn"]["population_size"]
    for id in range(population_size):
        p = mp.Process(target=run_gtn, args=(config, id))
        p.start()
        p_list.append(p)

    for p in p_list:
        p.join()




if __name__ == "__main__":
    with open("../default_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    # seed = config["seed"]
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    gtn = GTN(config)
    gtn.calc_noise_and_eps()

    # if args.purge:
    #     #
    #     gtn = GTN(config)
    #     gtn.purge()
    # else:
    #     id = args.id
    #     # execution on single PC
    #     if id < 0:
    #         # first clean working directory from old files
    #         gtn = GTN(config, args.id)
    #         gtn.purge()
    #         # then execute multiple threads
    #         run_gtn_on_single_pc(config)
    #
    #     # execution on cluster
    #     else:
    #         gtn = GTN(config, args.id)
    #         gtn.run()
