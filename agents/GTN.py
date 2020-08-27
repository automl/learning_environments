import argparse
import yaml
import torch
import torch.nn as nn
import multiprocessing as mp
import os
import time
import datetime
import uuid
import numpy as np
import glob
from agents.agent_utils import select_agent, test
from envs.env_factory import EnvFactory

# parser = argparse.ArgumentParser(description='GTN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('-p', '--purge', default=False, type=bool, help='purge workspace')
# parser.add_argument('-i', '--id', default=-1, type=int, help='id of the population member')
# args = parser.parse_args()



class GTN_Base(nn.Module):
    def __init__(self, config, id):
        super().__init__()

        # for saving/loading
        self.config = config

        gtn_config = config["agents"]["gtn"]
        self.max_iterations = gtn_config["max_iterations"]
        self.noise_std = gtn_config["noise_std"]
        self.minimize_score = gtn_config["minimize_score"]
        self.agent_name = gtn_config["agent_name"]
        self.device = config["device"]

        self.env_factory = EnvFactory(config)
        self.virtual_env_orig = self.env_factory.generate_virtual_env()

        # for identifying the different workers and the master
        self.id = id

        x = datetime.datetime.now()
        self.working_dir = str(os.path.join(os.getcwd(), "results", 'GTN_' + x.strftime("%Y-%m-%d-%H")))
        os.makedirs(self.working_dir, exist_ok=True)

    def get_input_file_name(self, id):
        return os.path.join(self.working_dir, str(id) + 'input_.pt')

    def get_result_file_name(self, id):
        return os.path.join(self.working_dir, str(id) + 'result_.pt')

    def clean_working_dir(self):
        files = glob.glob(os.path.join(self.working_dir, '*'))
        print(self.working_dir)
        for file in files:
            print(file)


class GTN_Master(GTN_Base):
    def __init__(self, config, id):
        super().__init__(config, id)

        gtn_config = config["agents"]["gtn"]
        self.num_workers = gtn_config["num_workers"]
        self.learning_rate = gtn_config["learning_rate"]
        self.score_transform_type = gtn_config["score_transform_type"]
        self.time_mult = gtn_config["time_mult"]
        self.time_max = gtn_config["time_max"]

        # id used as a handshake to check if resuls from workers correspond to sent data
        self.uuid_list = [0]*(self.population_size)

        # to store results from workers
        self.timing_list = [None]*self.num_workers
        self.score_list = [None]*self.num_workers
        self.eps_list = [self.env_factory.generate_virtual_env() for _ in range(self.num_workers)]

        # for early out
        self.avg_runtime = None


    def run(self):
        for it in range(self.max_iterations):
            self.write_worker_inputs(it)
            self.read_worker_results()
            self.rank_transform()
            self.update_env()


    def write_worker_inputs(self, it):
        for id in range(self.num_workers):
            file_name = self.get_input_file_name(id=id)

            # wait until worker has deleted the file (i.e. acknowledged the previous input)
            while os.path.isfile(file_name):
                time.sleep(0.001)

            self.uuid_list[id] = str(uuid.uuid4())
            quit_flag = it == self.max_iterations-1

            data = {}
            data['timeout'] = self.time_max  # todo: change for variable timeout
            data['uuid'] = self.uuid_list[id]
            data['quit_flag'] = quit_flag
            data['virtual_env_orig'] = self.virtual_env_orig.state_dict()


            torch.save(data, file_name)


    def read_worker_results(self):
        for id in range(self.num_workers):
            file_name = self.get_result_file_name(id)

            # wait until worker has finished calculations
            while not os.path.isfile(file_name):
                time.sleep(0.001)

            data = torch.load(file_name)
            os.remove(file_name)

            uuid = data['uuid']

            if uuid != self.uuid_list[id]:
                raise ValueError("UUIDs do not match")

            self.time_list[id] = data['time_elapsed']
            self.score_list[id] = data['score']
            self.eps_list[id] = data['eps']



    def rank_transform(self):
        scores = np.asarray(self.score_list)

        if self.minimize_score:
            scores = -scores

        if self.score_transform_type == 0:
            # convert [1, 0, 5] to [0.2, 0, 1]
            scores = (scores - min(scores)) / (max(scores)-min(scores))
        elif self.score_transform_type == 1:
            # convert [1, 0, 5] to [0.5, 0, 1]
            s = np.argsort(scores)
            n = len(scores)
            for i in range(n):
                scores[s[i]] = i / (n-1)
        elif self.score_transform_type == 2:
            # fitness shaping from from "Natural Evolution Strategies" (Wierstra 2014) paper
            lmbda = len(scores)
            s = np.argsort(-scores)
            for i in range(lmbda):
                scores[s[i]] = i + 1
            for i in range(lmbda):
                scores[i] = max(0, np.log(lmbda / 2 + 1) - np.log(scores[i]))
            scores = scores / sum(scores) - 1 / lmbda
            # additional normalization (not in original paper)
            scores /= max(scores)
        else:
            raise ValueError("Unknown rank transform type: " + str(self.rank_transform_type))

        self.score_list = scores.tolist()

    def update_env(self):
        n = self.num_workers
        lr = self.learning_rate
        sig = self.noise_std
        for eps, score in zip(self.eps_list, self.score_list):
            for l_orig, l_eps in zip(self.virtual_env_orig.modules(), self.eps.modules()):
                if isinstance(l_orig, nn.Linear):
                    l_orig.weight = torch.nn.Parameter(l_orig.weight + (lr/n/sig) * score * l_eps.weight )


class GTN_Worker(GTN_Base):
    def __init__(self, config, id):
        super().__init__(config, id)

        self.virtual_env = self.env_factory.generate_virtual_env()
        self.eps = self.env_factory.generate_virtual_env()
        self.uuid = None
        self.timeout = None
        self.quit_flag = False


    def run(self):
        # read data from master
        while not self.quit_flag:
            self.read_worker_input()

            time_start = time.time()

            agent_add = select_agent(config, self.agent_name)
            agent_sub = select_agent(config, self.agent_name)
            agent_sub.set_state_dict(agent_add.set_state_dict())

            self.get_random_eps()

            # first mirrored noise +N
            self.add_noise_to_virtual_env()
            agent_add.train(self.virtual_env, time_start=time_start, timeout=self.timeout)
            loss_add = self.test_agent_on_real_env(agent_add, time_start=time_start, timeout=self.timeout)

            # second mirrored noise +N
            self.subtract_noise_from_virtual_env()
            agent_sub.train(env=self.virtual_env, time_start=time_start, timeout=self.timeout)
            loss_sub = self.test_agent_on_real_env(agent_sub, time_start=time_start, timeout=self.timeout)

            if self.minimize_score:
                best_score = min(loss_add, loss_sub)
                if loss_add < loss_sub:
                    self.invert_eps()
            else:
                best_score = max(loss_add, loss_sub)
                if loss_add > loss_sub:
                    self.invert_eps()

            self.write_worker_result(score = best_score, time_elapsed = time.time()-time.start)


    def read_worker_input(self):
        file_name = self.get_input_file_name(id=self.id)

        while not os.path.isfile(file_name):
            time.sleep(0.001)

        data = torch.load(file_name)
        os.remove(file_name)

        self.virtual_env_orig.load_state_dict(data['virtual_env_orig'])
        self.uuid = data['uuid']
        self.timeout = data['timeout']
        self.quit_flag = data['quit_flag']

    def write_worker_result(self, score, time_elapsed):
        file_name = self.get_result_file_name(id=self.id)

        while os.path.isfile(file_name):
            time.sleep(0.001)

        data = {}
        data["eps"] = self.virtual_env.state_dict()
        data["time_elapsed"] = time_elapsed
        data["score"] = score
        torch.save(data, file_name)


    def get_random_eps(self):
        for l_virt, l_eps in zip(self.virtual_env.modules(), self.eps.modules()):
            if isinstance(l_virt, nn.Linear):
                l_eps.weight = torch.nn.Parameter(torch.normal(mean=torch.zeros_like(l_virt.weight),
                                                               std=torch.ones_like(l_virt.weight)))

    def add_noise_to_virtual_env(self, add=True):
        for l_orig, l_virt, l_eps in zip(self.virtual_env_orig.modules(), self.virtual_env.modules(), self.eps.modules()):
            if isinstance(l_virt, nn.Linear):
                if add: # add eps
                    l_virt.weight = torch.nn.Parameter(l_orig.weight + l_eps.weight * self.noise_std)
                else:   # subtract eps
                    l_virt.weight = torch.nn.Parameter(l_orig.weight - l_eps.weight * self.noise_std)

    def subtract_noise_from_virtual_env(self):
        self.add_noise_to_virtual_env(add=False)

    def invert_eps(self):
        for l_eps in self.eps.modules():
            if isinstance(l_eps, nn.Linear):
                l_eps.weight = torch.nn.Parameter(-l_eps.weight)

    def test_agent_on_real_env(self, agent, time_start, timeout):
        mean_episodes_till_solved, episodes_till_solved = test(agent=agent,
                                                               env_factory=self.env_factory,
                                                               config=self.config,
                                                               time_start=time_start,
                                                               timeout=timeout)
        return mean_episodes_till_solved


def run_gtn_on_single_pc(config):
    def run_gtn_worker(config, id):
        gtn = GTN_Worker(config, id)
        gtn.run()

    def run_gtn_master(config):
        gtn = GTN_Master(config, -1)
        gtn.run()

    p_list = []

    # cleanup working directory from old files
    gtn_base = GTN_Base
    gtn_base.clean_working_dir()

    # first start master
    p = mp.Process(target=run_gtn_master, args=(config, -1))
    p.start()
    p_list.append(p)

    # then start workers
    population_size = config["gtn"]["population_size"]
    for id in range(population_size):
        p = mp.Process(target=run_gtn_worker, args=(config, id))
        p.start()
        p_list.append(p)

    # wait till everything has finished
    for p in p_list:
        p.join()




if __name__ == "__main__":
    with open("../default_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    run_gtn_on_single_pc(config)

    # seed = config["seed"]
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    # gtn.write_result_to_file(loss=0.1, time_elapsed=10, it=5)
    # gtn.read_result_from_file(it=5, id=2)

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
