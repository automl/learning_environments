import yaml
import torch
import torch.nn as nn
import multiprocessing as mp
import os
import sys
import time
import datetime
import uuid
import numpy as np
import glob
import statistics
from utils import calc_abs_param_sum, print_abs_param_sum
from agents.agent_utils import select_agent, test
from envs.env_factory import EnvFactory

# parser = argparse.ArgumentParser(description='GTN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('-p', '--purge', default=False, type=bool, help='purge workspace')
# parser.add_argument('-i', '--id', default=-1, type=int, help='id of the population member')
# args = parser.parse_args()



class GTN_Base(nn.Module):
    def __init__(self, config):
        super().__init__()

        # for saving/loading
        self.config = config

        gtn_config = config["agents"]["gtn"]
        self.max_iterations = gtn_config["max_iterations"]
        self.noise_std = gtn_config["noise_std"]
        self.time_sleep = gtn_config["time_sleep"]
        self.minimize_score = gtn_config["minimize_score"]
        self.agent_name = gtn_config["agent_name"]
        self.device = config["device"]

        self.env_factory = EnvFactory(config)
        self.virtual_env_orig = self.env_factory.generate_virtual_env(print_str='GTN_Base: ')

        x = datetime.datetime.now()
        self.working_dir = str(os.path.join(os.getcwd(), "results", 'GTN_' + x.strftime("%Y-%m-%d-%H")))
        os.makedirs(self.working_dir, exist_ok=True)

    def get_input_file_name(self, id):
        return os.path.join(self.working_dir, str(id) + '_input.pt')

    def get_input_check_file_name(self, id):
        return os.path.join(self.working_dir, str(id) + '_input_check.pt')

    def get_result_file_name(self, id):
        return os.path.join(self.working_dir, str(id) + '_result.pt')

    def get_result_check_file_name(self, id):
        return os.path.join(self.working_dir, str(id) + '_result_check.pt')

    def clean_working_dir(self):
        files = glob.glob(os.path.join(self.working_dir, '*'))
        for file in files:
            os.remove(file)


class GTN_Master(GTN_Base):
    def __init__(self, config):
        super().__init__(config)

        gtn_config = config["agents"]["gtn"]
        self.num_workers = gtn_config["num_workers"]
        self.learning_rate = gtn_config["learning_rate"]
        self.score_transform_type = gtn_config["score_transform_type"]
        self.time_mult = gtn_config["time_mult"]
        self.time_max = gtn_config["time_max"]

        # id used as a handshake to check if resuls from workers correspond to sent data
        self.uuid_list = [0]*(self.num_workers)

        # to store results from workers
        self.time_list = [None]*self.num_workers
        self.score_list = [None]*self.num_workers
        self.score_transform_list = [None]*self.num_workers
        self.eps_list = [self.env_factory.generate_virtual_env(print_str='GTN_Master: ') for _ in range(self.num_workers)]

        # for early out
        self.avg_runtime = None


    def run(self):
        for it in range(self.max_iterations):
            print('-- Master: start iteration ' + str(it))
            print('-- Master: write worker inputs')
            self.write_worker_inputs(it)
            print('-- Master: read worker results')
            self.read_worker_results()
            print('-- Master: rank transform')
            self.rank_transform()
            print('-- Master: update env')
            self.update_env()
            print('-- Master: print statistics')
            self.print_statistics(it)

        print('Master quitting')



    def calc_worker_timeout(self):
        if self.time_list[0] is None:
            return self.time_max
        else:
            return statistics.mean(self.time_list) * self.time_mult


    def write_worker_inputs(self, it):
        timeout = self.calc_worker_timeout()
        print('timeout: ' + str(timeout) + " " + str(self.time_list))

        for id in range(self.num_workers):
            file_name = self.get_input_file_name(id=id)
            check_file_name = self.get_input_check_file_name(id=id)

            # wait until worker has deleted the file (i.e. acknowledged the previous input)
            while os.path.isfile(file_name):
                time.sleep(self.time_sleep)

            self.uuid_list[id] = str(uuid.uuid4())
            quit_flag = it == self.max_iterations-1

            data = {}
            data['timeout'] = timeout
            data['uuid'] = self.uuid_list[id]
            data['quit_flag'] = quit_flag
            data['virtual_env_orig'] = self.virtual_env_orig.state_dict()

            torch.save(data, file_name)
            torch.save({}, check_file_name)


    def read_worker_results(self):
        for id in range(self.num_workers):
            file_name = self.get_result_file_name(id)
            check_file_name = self.get_result_check_file_name(id)

            # wait until worker has finished calculations
            while not os.path.isfile(check_file_name):
                time.sleep(self.time_sleep)

            data = torch.load(file_name)

            uuid = data['uuid']

            if uuid != self.uuid_list[id]:
                raise ValueError("UUIDs do not match")

            self.time_list[id] = data['time_elapsed']
            self.score_list[id] = data['score']
            self.eps_list[id].load_state_dict(data['eps'])

            os.remove(check_file_name)
            os.remove(file_name)


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
            scores = scores.astype(float)
            for i in range(lmbda):
                scores[i] = max(0, np.log(lmbda / 2 + 1) - np.log(scores[i]))
            scores = scores / sum(scores) - 1 / lmbda
            # additional normalization (not in original paper)
            scores /= max(scores)
        else:
            raise ValueError("Unknown rank transform type: " + str(self.rank_transform_type))

        self.score_transform_list = scores.tolist()


    def update_env(self):
        n = self.num_workers
        lr = self.learning_rate
        sig = self.noise_std
        print('-- update env --')
        print(self.score_list)
        print(self.score_transform_list)
        print_abs_param_sum(self.virtual_env_orig)

        for eps, score_transform in zip(self.eps_list, self.score_transform_list):
            for l_orig, l_eps in zip(self.virtual_env_orig.modules(), eps.modules()):
                if isinstance(l_orig, nn.Linear):
                    l_orig.weight = torch.nn.Parameter(l_orig.weight + (lr/n/sig) * score_transform * l_eps.weight )

        print_abs_param_sum(self.virtual_env_orig)


    def print_statistics(self, it):
        mean_score = statistics.mean(self.score_list)
        print('--------------')
        print('GTN iteration:  ' + str(it))
        print('GTN mean score: ' + str(mean_score))
        print('--------------')


class GTN_Worker(GTN_Base):
    def __init__(self, config, id):
        super().__init__(config)
        torch.manual_seed(id+int(time.time()))

        gtn_config = config["agents"]["gtn"]
        self.num_test_envs = gtn_config["num_test_envs"]
        self.virtual_env = self.env_factory.generate_virtual_env(print_str='GTN_Worker' + str(id) + ': ')
        self.eps = self.env_factory.generate_virtual_env('GTN_Worker' + str(id) + ': ')
        self.uuid = None
        self.timeout = None
        self.quit_flag = False

        # for identifying the different workers
        self.id = id


    def run(self):
        # read data from master
        while not self.quit_flag:
            print('-- Worker {}: read worker inputs'.format(self.id))
            self.read_worker_input()

            print('-- Worker {}: precalculation'.format(self.id))

            time_start = time.time()

            agent_add = select_agent(config, self.agent_name)
            agent_sub = select_agent(config, self.agent_name)
            agent_sub.load_state_dict(agent_add.state_dict())

            self.get_random_eps()

            print('-- Worker {}: train add'.format(self.id))

            # first mirrored noise +N
            self.add_noise_to_virtual_env()
            #print('{} {} {} {}'.format(self.id,
                                       # calc_abs_param_sum(self.virtual_env_orig),
                                       # calc_abs_param_sum(self.virtual_env),
                                       # calc_abs_param_sum(self.eps)))
            agent_add.train(self.virtual_env, time_remaining=self.timeout-(time.time()-time_start))
            score_add = self.test_agent_on_real_env(agent_add, time_remaining=self.timeout-(time.time()-time_start))

            print('-- Worker {}: train sub'.format(self.id))

            # second mirrored noise +N
            self.subtract_noise_from_virtual_env()
            # print('{} {} {} {}'.format(self.id,
            #                            calc_abs_param_sum(self.virtual_env_orig),
            #                            calc_abs_param_sum(self.virtual_env),
            #                            calc_abs_param_sum(self.eps)))
            agent_sub.train(env=self.virtual_env, time_remaining=self.timeout-(time.time()-time_start))
            score_sub = self.test_agent_on_real_env(agent_sub, time_remaining=self.timeout-(time.time()-time_start))

            print('-- Worker {}: postcalculation'.format(self.id))

            if self.minimize_score:
                best_score = min(score_add, score_sub)
                if score_sub < score_add:
                    self.invert_eps()
            else:
                best_score = max(score_add, score_sub)
                if score_sub > score_add:
                    self.invert_eps()

            print('-- LOSS ADD: ' + str(score_add))
            print('-- LOSS SUB: ' + str(score_sub))
            print('-- LOSS BEST: ' + str(best_score))
            print('-- Worker {}: write result'.format(self.id))

            self.write_worker_result(score = best_score, time_elapsed = time.time()-time_start)

        print('Agent ' + str(self.id) + ' quitting')

    def read_worker_input(self):
        file_name = self.get_input_file_name(id=self.id)
        check_file_name = self.get_input_check_file_name(id=self.id)

        while not os.path.isfile(check_file_name):
            time.sleep(self.time_sleep)

        data = torch.load(file_name)

        self.virtual_env_orig.load_state_dict(data['virtual_env_orig'])
        self.uuid = data['uuid']
        self.timeout = data['timeout']
        self.quit_flag = data['quit_flag']

        os.remove(check_file_name)
        os.remove(file_name)


    def write_worker_result(self, score, time_elapsed):
        file_name = self.get_result_file_name(id=self.id)
        check_file_name = self.get_result_check_file_name(id=self.id)

        # wait until master has deleted the file (i.e. acknowledged the previous result)
        while os.path.isfile(file_name):
            time.sleep(self.time_sleep)

        data = {}
        data["eps"] = self.virtual_env.state_dict()
        data["time_elapsed"] = time_elapsed
        data["score"] = score
        data["uuid"] = self.uuid
        torch.save(data, file_name)
        torch.save({}, check_file_name)


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


    def test_agent_on_real_env(self, agent, time_remaining):
        mean_episodes_till_solved, episodes_till_solved = test(agent=agent,
                                                               env_factory=self.env_factory,
                                                               config=self.config,
                                                               num_envs=self.num_test_envs,
                                                               time_remaining=time_remaining)
        return mean_episodes_till_solved


def run_gtn_on_single_pc(config):
    def run_gtn_worker(config, id):
        gtn = GTN_Worker(config, id)
        gtn.run()

    def run_gtn_master(config):
        gtn = GTN_Master(config)
        gtn.run()

    p_list = []

    # cleanup working directory from old files
    gtn_base = GTN_Base(config)
    gtn_base.clean_working_dir()
    time.sleep(2)

    # first start master
    p = mp.Process(target=run_gtn_master, args=(config,))
    p.start()
    p_list.append(p)

    # then start workers
    num_workers = config["agents"]["gtn"]["num_workers"]
    for id in range(num_workers):
        p = mp.Process(target=run_gtn_worker, args=(config, id))
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
        gtn_worker = GTN_Worker(config, id)
        gtn_worker.run()
    else:
        raise ValueError("Invalid ID")

if __name__ == "__main__":
    with open("../default_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    gtn_config = config['agents']['gtn']
    mode = gtn_config['mode']

    torch.set_num_threads(gtn_config['num_threads_per_worker'])

    if mode == 'single':
        run_gtn_on_single_pc(config)
    elif mode == 'multi':
        for arg in sys.argv[1:]:
            print(arg)

        id = int(sys.argv[1])
        run_gtn_on_multiple_pcs(config, id)

