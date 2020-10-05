import yaml
import torch
import torch.nn as nn
import multiprocessing as mp
import os
import sys
import time
import uuid
import numpy as np
import math
import glob
import random
import string
import statistics
from collections import Counter
from utils import calc_abs_param_sum, ReplayBuffer, print_abs_param_sum, to_one_hot_encoding, from_one_hot_encoding
from agents.agent_utils import select_agent
from agents.env_matcher import EnvMatcher
from envs.env_factory import EnvFactory



class GTN_Base(nn.Module):
    def __init__(self, config, bohb_id):
        super().__init__()

        # for saving/loading
        self.config = config

        gtn_config = config["agents"]["gtn"]
        self.max_iterations = gtn_config["max_iterations"]
        self.minimize_score = gtn_config["minimize_score"]
        self.match = gtn_config["match"]
        self.rb_size = gtn_config["rb_size"]
        self.agent_name = gtn_config["agent_name"]
        self.device = config["device"]

        self.bohb_id = bohb_id

        self.env_factory = EnvFactory(config)
        self.virtual_env_orig = self.env_factory.generate_virtual_env(print_str='GTN_Base: ')

        self.working_dir = str(os.path.join(os.getcwd(), "results", 'GTN_sync__2x3'))

        os.makedirs(self.working_dir, exist_ok=True)

    def get_input_file_name(self, id):
        return os.path.join(self.working_dir, str(self.bohb_id) + '_' + str(id) + '_input.pt')

    def get_input_check_file_name(self, id):
        return os.path.join(self.working_dir, str(self.bohb_id) + '_' + str(id) + '_input_check.pt')

    def get_result_file_name(self, id):
        return os.path.join(self.working_dir, str(self.bohb_id) + '_' + str(id) + '_result.pt')

    def get_result_check_file_name(self, id):
        return os.path.join(self.working_dir, str(self.bohb_id) + '_' + str(id) + '_result_check.pt')

    def clean_working_dir(self):
        files = glob.glob(os.path.join(self.working_dir, '*'))
        for file in files:
            os.remove(file)

    def create_replay_buffer(self):
        sd = 1 if self.virtual_env_orig.has_discrete_state_space() else self.state_dim
        ad = 1 if self.virtual_env_orig.has_discrete_action_space() else self.action_dim
        return ReplayBuffer(state_dim=sd, action_dim=ad, device=self.device, max_size=self.rb_size)


class GTN_Master(GTN_Base):
    def __init__(self, config, bohb_id=-1):
        super().__init__(config, bohb_id)

        gtn_config = config["agents"]["gtn"]
        self.num_workers = gtn_config["num_workers"]
        self.step_size = gtn_config["step_size"]
        self.weight_decay = gtn_config["weight_decay"]
        self.score_transform_type = gtn_config["score_transform_type"]
        self.time_mult = gtn_config["time_mult"]
        self.time_max = gtn_config["time_max"]
        self.time_sleep_master = gtn_config["time_sleep_master"]

        # id used as a handshake to check if resuls from workers correspond to sent data
        self.uuid_list = [0]*(self.num_workers)

        # to store results from workers
        self.time_train_list = [None]*self.num_workers # for debugging
        self.time_test_list = [None] * self.num_workers # for debugging
        self.time_elapsed_list = [None] * self.num_workers # for debugging
        self.score_list = [None]*self.num_workers
        self.score_orig_list = [None]*self.num_workers # for debugging
        self.score_transform_list = [None]*self.num_workers
        self.virtual_env_list = [self.env_factory.generate_virtual_env(print_str='GTN_Master: ') for _ in range(self.num_workers)]
        self.eps_list = [self.env_factory.generate_virtual_env(print_str='GTN_Master: ') for _ in range(self.num_workers)]

        # for early out
        self.avg_runtime = None
        self.real_env = self.env_factory.generate_default_real_env()
        self.optimal_path = self.find_optimal_path()

        # to store models
        self.model_dir = str(os.path.join(os.getcwd(), "results", 'GTN_models'))
        # for matching
        self.rb = self.create_replay_buffer()

        os.makedirs(self.model_dir, exist_ok=True)

        print('Starting GTN Master with bohb_id {}'.format(bohb_id))
        print('optimal path: {}'.format(self.optimal_path))


    def get_model_file_name(self, file_name):
        return os.path.join(self.model_dir, file_name)


    def find_optimal_path(self,):
        agent_base = select_agent(config=self.config, agent_name=self.agent_name)
        # agent_base.train(env=self.virtual_env_orig,
        #                  time_remaining=1e3)
        agent_base.train(env=self.real_env,
                         time_remaining=1e3)

        agent_base.test_episodes = 1
        _, replay_buffer = agent_base.test(env=self.real_env)
        _, _, next_state, _, _ = replay_buffer.get_all()
        optimal_path = [state.item() for state in next_state.int()]

        return optimal_path


    def run(self):
        mean_score_orig_list = []
        model_saved = False

        for it in range(self.max_iterations):
            t1 = time.time()
            print('-- Master: Iteration ' + str(it) + ' ' + str(time.time()-t1))
            #print('-- Master: start iteration ' + str(it))
            print('-- Master: write worker inputs' + ' ' + str(time.time()-t1))
            self.write_worker_inputs(it)
            print('-- Master: read worker results' + ' ' + str(time.time()-t1))
            skip_flag = self.read_worker_results()
            if skip_flag:
                continue
            print('-- Master: rank transform' + ' ' + str(time.time()-t1))
            self.score_transform()
            print('-- Master: update env' + ' ' + str(time.time()-t1))
            self.update_env()
            print('-- Master: match env' + ' ' + str(time.time()-t1))
            self.match_env()
            print('-- Master: print statistics' + ' ' + str(time.time()-t1))
            self.print_statistics(it=it, time_elapsed=time.time()-t1)

            mean_score_orig_list.append(np.mean(self.score_orig_list))

            if np.mean(self.score_orig_list) > self.real_env.get_solved_reward() and not model_saved:
                self.save_good_model(mean_score_orig_list)
                model_saved = True
            #     break

        print('Master quitting')

        self.save_good_model(mean_score_orig_list)

        # error handling
        if len(mean_score_orig_list) > 0:
            return np.mean(self.score_orig_list), mean_score_orig_list
        else:
            return 1e9, mean_score_orig_list


    def save_good_model(self, mean_score_orig_list):
        it = len(mean_score_orig_list)
        if it > 0 and it < self.max_iterations-1:
            random_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 6))
            m = len(self.real_env.env.grid)
            n = len(self.real_env.env.grid[0])
            file_name = self.get_model_file_name(str(m) + 'x' + str(n) + '_' + str(it) + '_' + random_string + '.pt')
            save_dict = {}
            save_dict['model'] = self.virtual_env_orig.state_dict()
            save_dict['config'] = self.config
            torch.save(save_dict, file_name)


    def calc_worker_timeout(self):
        if self.time_elapsed_list[0] is None:
            return self.time_max
        else:
            return statistics.mean(self.time_elapsed_list) * self.time_mult


    def write_worker_inputs(self, it):
        timeout = self.calc_worker_timeout()
        print('timeout: ' + str(timeout))

        for id in range(self.num_workers):
            file_name = self.get_input_file_name(id=id)
            check_file_name = self.get_input_check_file_name(id=id)

            # wait until worker has deleted the file (i.e. acknowledged the previous input)
            while os.path.isfile(file_name):
                time.sleep(self.time_sleep_master)

            time.sleep(self.time_sleep_master)
            self.uuid_list[id] = str(uuid.uuid4())

            # if we are not using bohb, shut everything down after last iteration
            if self.bohb_id < 0:
                quit_flag = it == self.max_iterations-1
            else:
                quit_flag = False

            data = {}
            data['timeout'] = timeout
            data['uuid'] = self.uuid_list[id]
            data['quit_flag'] = quit_flag
            data['gtn_iteration'] = it
            data['optimal_path'] = self.optimal_path
            data['virtual_env_orig'] = self.virtual_env_orig.state_dict()

            torch.save(data, file_name)
            torch.save({}, check_file_name)


    def read_worker_results(self):
        skip_flag = False

        for id in range(self.num_workers):
            file_name = self.get_result_file_name(id)
            check_file_name = self.get_result_check_file_name(id)

            # wait until worker has finished calculations
            while not os.path.isfile(check_file_name):
                time.sleep(self.time_sleep_master)

            data = torch.load(file_name)

            uuid = data['uuid']

            if uuid != self.uuid_list[id]:
                skip_flag = True
                print("UUIDs do not match")

                input_file_name = self.get_input_file_name(id=id)
                input_check_file_name = self.get_input_check_file_name(id=id)

                if os.path.isfile(input_file_name):
                    os.remove(input_file_name)

                if os.path.isfile(input_file_name):
                    os.remove(input_check_file_name)

            self.time_train_list[id] = data['time_train']
            self.time_test_list[id] = data['time_test']
            self.time_elapsed_list[id] = data['time_elapsed']
            self.score_list[id] = data['score']
            self.eps_list[id].load_state_dict(data['eps'])
            self.score_orig_list[id] = data['score_orig']                  # for debugging
            self.virtual_env_list[id].load_state_dict(data['virtual_env']) # for debugging

            self.rb.merge_vectors(states=data['rb_states'],
                                  actions=data['rb_actions'],
                                  next_states=data['rb_next_states'],
                                  rewards=data['rb_rewards'],
                                  dones=data['rb_dones'])

            os.remove(check_file_name)
            os.remove(file_name)

        return skip_flag

    def score_transform(self):
        scores = np.asarray(self.score_list)
        scores_orig = np.asarray(self.score_orig_list)

        if self.minimize_score:
            scores = -scores
            scores_orig = -scores_orig

        if self.score_transform_type == 0:
            # convert [1, 0, 5] to [0.2, 0, 1]
            scores = (scores - min(scores)) / (max(scores)-min(scores)+1e-9)

        elif self.score_transform_type == 1:
            # convert [1, 0, 5] to [0.5, 0, 1]
            s = np.argsort(scores)
            n = len(scores)
            for i in range(n):
                scores[s[i]] = i / (n-1)

        elif self.score_transform_type == 2 or self.score_transform_type == 3:
            # fitness shaping from "Natural Evolution Strategies" (Wierstra 2014) paper, either with zero mean (2) or without (3)
            lmbda = len(scores)
            s = np.argsort(-scores)
            for i in range(lmbda):
                scores[s[i]] = i + 1
            scores = scores.astype(float)
            for i in range(lmbda):
                scores[i] = max(0, np.log(lmbda / 2 + 1) - np.log(scores[i]))

            scores = scores / sum(scores)

            if self.score_transform_type == 2:
                scores -= 1 / lmbda

            scores /= max(scores)

        elif self.score_transform_type == 4:
            # consider single best eps
            scores_tmp = np.zeros(scores.size)
            scores_tmp[np.argmax(scores)] = 1
            scores = scores_tmp
        elif self.score_transform_type == 5:
            # consider all eps that are better than the average
            avg_score_orig = np.mean(scores_orig)

            scores_idx = np.where(scores > avg_score_orig + 1e-6,1,0)   # 1e-6 to counter numerical errors
            if sum(scores_idx) > 0:
            #if sum(scores_idx) > 0:
                scores = scores_idx * (scores-avg_score_orig) / (max(scores)-avg_score_orig+1e-9)
                scores /= sum(scores)
            else:
                scores = scores_idx

        elif self.score_transform_type == 6:
            # consider single best eps that is better than the average
            avg_score_orig = np.mean(scores_orig)

            scores_idx = np.where(scores > avg_score_orig + 1e-6,1,0)   # 1e-6 to counter numerical errors
            if sum(scores_idx) > 0:
                scores_tmp = np.zeros(scores.size)
                scores_tmp[np.argmax(scores)] = 1
                scores = scores_tmp
            else:
                scores = scores_idx
        else:
            raise ValueError("Unknown rank transform type: " + str(self.score_transform_type))

        self.score_transform_list = scores.tolist()


    def update_env(self):
        ss = self.step_size
        # print('-- update env --')
        print('score_orig_list      ' + str(self.score_orig_list))
        print('score_list           ' + str(self.score_list))
        print('score_transform_list ' + str(self.score_transform_list))
        #print('venv weights         ' + str([calc_abs_param_sum(elem).item() for elem in self.virtual_env_list]))
        #print('mean                 ' + str(np.mean(self.score_transform_list)))

        print('weights before: ' + str(calc_abs_param_sum(self.virtual_env_orig).item()))

        # weight decay
        for l_orig in self.virtual_env_orig.modules():
            if isinstance(l_orig, nn.Linear):
                l_orig.weight = torch.nn.Parameter(l_orig.weight * (1 - self.weight_decay))
                l_orig.bias = torch.nn.Parameter(l_orig.bias * (1 - self.weight_decay))

        print('weights weight decay: ' + str(calc_abs_param_sum(self.virtual_env_orig).item()))

        # weight update
        for eps, score_transform in zip(self.eps_list, self.score_transform_list):
            for l_orig, l_eps in zip(self.virtual_env_orig.modules(), eps.modules()):
                if isinstance(l_orig, nn.Linear):
                    l_orig.weight = torch.nn.Parameter(l_orig.weight + ss * score_transform * l_eps.weight)
                    l_orig.bias = torch.nn.Parameter(l_orig.bias + ss * score_transform * l_eps.bias)

        print('weights update: ' + str(calc_abs_param_sum(self.virtual_env_orig).item()))


    def match_env(self):
        if not self.match:
            return

        me = EnvMatcher(config)
        me.train(virtual_env=self.virtual_env_orig, replay_buffer=self.rb)

        states, _, next_states, _, _ = self.rb.get_all()

        next_states = [state.item() for state in next_states.int()]
        print('-- next_states -- ' + str(Counter(next_states)) + ' ' + str(len(next_states)))

        agent = select_agent(config, 'QL')
        avg_reward = agent.test(env=self.real_env)
        print('-- avg_reward -- ' + str(avg_reward))

    def print_statistics(self, it, time_elapsed):
        orig_score = statistics.mean(self.score_orig_list)
        #dist_score = statistics.mean(self.score_list)
        mean_time_train = statistics.mean(self.time_train_list)
        mean_time_test = statistics.mean(self.time_test_list)
        mean_time_elapsed = statistics.mean(self.time_elapsed_list)
        print('--------------')
        print('GTN iteration:    ' + str(it))
        print('GTN mstr t_elaps: ' + str(time_elapsed))
        print('GTN avg wo t_elaps: ' + str(mean_time_elapsed))
        print('GTN avg wo t_train: ' + str(mean_time_train))
        print('GTN avg wo t_test:  ' + str(mean_time_test))
        #print('GTN avg dist score:   ' + str(dist_score))
        print('GTN best dist score:  ' + str(min(self.score_list)))
        print('GTN avg eval score:   ' + str(orig_score))
        print('--------------')


class GTN_Worker(GTN_Base):
    def __init__(self, config, id, bohb_id=-1):
        super().__init__(config, bohb_id)
        torch.manual_seed(id+int(time.time()))

        gtn_config = config["agents"]["gtn"]
        self.noise_std = gtn_config["noise_std"]
        self.num_test_envs = gtn_config["num_test_envs"]
        self.num_grad_evals = gtn_config["num_grad_evals"]
        self.grad_eval_type = gtn_config["grad_eval_type"]
        self.exploration_gain = gtn_config["exploration_gain"]
        self.correct_path_gain = gtn_config["correct_path_gain"]
        self.time_sleep_worker = gtn_config["time_sleep_worker"]
        self.virtual_env = self.env_factory.generate_virtual_env(print_str='GTN_Worker' + str(id) + ': ')
        self.eps = self.env_factory.generate_virtual_env('GTN_Worker' + str(id) + ': ')
        self.gtn_iteration = None
        self.uuid = None
        self.timeout = None
        self.quit_flag = False
        self.r_states = []
        self.r_actions = []

        # for identifying the different workers
        self.id = id

        print('Starting GTN Worker with bohb_id {} and id {}'.format(bohb_id, id))


    def run(self):
        # read data from master
        while not self.quit_flag:
            t1 = time.time()

            rb_all = self.create_replay_buffer()

            print('-- Worker {}: read worker inputs {}'.format(self.id, time.time()-t1))

            self.read_worker_input()

            print('-- Worker {}: evaluation {}'.format(self.id, time.time()-t1))

            time_start = time.time()

            # for evaluation purpose
            agent_orig = select_agent(config=self.config,
                                      agent_name=self.agent_name)
            tt1 = time.time()
            #print_abs_param_sum(self.virtual_env_orig)
            #print('-- Worker {}: ev train'.format(self.id))
            agent_orig.train(env=self.virtual_env_orig,
                             time_remaining=self.timeout-(time.time()-time_start),
                             gtn_iteration=self.gtn_iteration)
            tt2 = time.time()
            #print('-- Worker {}: ev test'.format(self.id))
            score_orig, rb = self.test_agent_on_real_env(agent=agent_orig,
                                                         time_remaining=self.timeout-(time.time()-time_start),
                                                         gtn_iteration=self.gtn_iteration)
            if self.match:
                rb_all.merge_buffer(rb)
            tt3 = time.time()
            time_train = tt2-tt1
            time_test = tt3-tt2

            #print_abs_param_sum(self.virtual_env_orig)
            #print(score_orig)

            self.get_random_eps()

            print('-- Worker {}: train add {}'.format(self.id, time.time()-t1))

            # first mirrored noise +N
            self.add_noise_to_virtual_env()

            score_add = []
            for i in range(self.num_grad_evals):
                #print('add ' + str(i))
                agent_add = select_agent(config=self.config,
                                         agent_name=self.agent_name)
                agent_add.train(env=self.virtual_env,
                                time_remaining=self.timeout-(time.time()-time_start),
                                gtn_iteration=self.gtn_iteration)
                score, rb = self.test_agent_on_real_env(agent=agent_add,
                                                        time_remaining=self.timeout-(time.time()-time_start),
                                                        gtn_iteration=self.gtn_iteration)
                score_add.append(score)
                if self.match:
                    rb_all.merge_buffer(rb)

            print('-- Worker {}: train sub {}'.format(self.id, time.time()-t1))

            # # second mirrored noise -N
            self.subtract_noise_from_virtual_env()

            score_sub = []
            for i in range(self.num_grad_evals):
                #print('sub ' + str(i))
                agent_sub = select_agent(config=self.config,
                                         agent_name=self.agent_name)
                agent_sub.train(env=self.virtual_env,
                                time_remaining=self.timeout-(time.time()-time_start),
                                gtn_iteration=self.gtn_iteration)
                score, rb = self.test_agent_on_real_env(agent=agent_sub,
                                                        time_remaining=self.timeout-(time.time()-time_start),
                                                        gtn_iteration=self.gtn_iteration)
                score_sub.append(score)
                if self.match:
                    rb_all.merge_buffer(rb)

            print('-- Worker {}: calc score {}'.format(self.id, time.time()-t1))

            if self.minimize_score:
                #print('worker ' + str(self.id) + ' ' + str(score_sub) + ' ' + str(score_add) + ' ' + str(weight_sub) + ' ' + str(weight_add))
                if self.grad_eval_type == 'mean':
                    score_sub = statistics.mean(score_sub)
                    score_add = statistics.mean(score_add)
                elif self.grad_eval_type == 'minmax':
                    score_sub = max(score_sub)
                    score_add = max(score_add)
                else:
                    raise NotImplementedError('Unknown parameter for grad_eval_type: ' + str(self.grad_eval_type))
                best_score = min(score_add, score_sub)
                if score_sub < score_add:
                    self.invert_eps()
                else:
                    self.add_noise_to_virtual_env() # for debugging
            else:
                if self.grad_eval_type == 'mean':
                    score_sub = statistics.mean(score_sub)
                    score_add = statistics.mean(score_add)
                elif self.grad_eval_type == 'minmax':
                    score_sub = min(score_sub)
                    score_add = min(score_add)
                else:
                    raise NotImplementedError('Unknown parameter for grad_eval_type: ' + str(self.grad_eval_type))
                best_score = max(score_add, score_sub)
                if score_sub > score_add:
                    self.invert_eps()
                else:
                    self.add_noise_to_virtual_env() # for debugging

            # print('-- LOSS ADD: ' + str(score_add))
            # print('-- LOSS SUB: ' + str(score_sub))
            # print('-- LOSS BEST: ' + str(best_score))
            print('-- Worker {}: write result '.format(self.id, time.time()-t1))

            self.write_worker_result(score=best_score,
                                     score_orig=score_orig,
                                     time_train=time_train,
                                     time_test=time_test,
                                     time_elapsed = time.time()-time_start,
                                     replay_buffer=rb_all)

        print('Worker ' + str(self.id) + ' quitting')

    def read_worker_input(self):
        file_name = self.get_input_file_name(id=self.id)
        check_file_name = self.get_input_check_file_name(id=self.id)

        while not os.path.isfile(check_file_name):
            time.sleep(self.time_sleep_worker)
        time.sleep(self.time_sleep_worker)

        data = torch.load(file_name)

        self.virtual_env_orig.load_state_dict(data['virtual_env_orig'])
        self.virtual_env.load_state_dict(data['virtual_env_orig'])
        self.uuid = data['uuid']
        self.gtn_iteration = data['gtn_iteration']
        self.timeout = data['timeout']
        self.quit_flag = data['quit_flag']
        self.optimal_path = data['optimal_path']

        os.remove(check_file_name)
        os.remove(file_name)


    def write_worker_result(self, score, score_orig, time_train, time_test, time_elapsed, replay_buffer):
        file_name = self.get_result_file_name(id=self.id)
        check_file_name = self.get_result_check_file_name(id=self.id)

        # wait until master has deleted the file (i.e. acknowledged the previous result)
        while os.path.isfile(file_name):
            time.sleep(self.time_sleep_worker)

        states, actions, next_states, rewards, dones = replay_buffer.get_all()

        data = {}
        data["eps"] = self.eps.state_dict()
        data["virtual_env"] = self.virtual_env.state_dict() # for debugging
        data["time_train"] = time_train
        data["time_test"] = time_test
        data["time_elapsed"] = time_elapsed # for debugging
        data["score"] = score
        data["score_orig"] = score_orig
        data["uuid"] = self.uuid
        data['rb_states'] = states
        data['rb_actions'] = actions
        data['rb_next_states'] = next_states
        data['rb_rewards'] = rewards
        data['rb_dones'] = dones
        torch.save(data, file_name)
        torch.save({}, check_file_name)


    def get_random_eps(self):
        #print(self.virtual_env)
        for l_virt, l_eps in zip(self.virtual_env.modules(), self.eps.modules()):
            if isinstance(l_virt, nn.Linear):
                l_eps.weight = torch.nn.Parameter(torch.normal(mean=torch.zeros_like(l_virt.weight),
                                                               std=torch.ones_like(l_virt.weight)) * self.noise_std)
                l_eps.bias = torch.nn.Parameter(torch.normal(mean=torch.zeros_like(l_virt.bias),
                                                               std=torch.ones_like(l_virt.bias)) * self.noise_std)


    def add_noise_to_virtual_env(self, add=True):
        for l_orig, l_virt, l_eps in zip(self.virtual_env_orig.modules(), self.virtual_env.modules(), self.eps.modules()):
            if isinstance(l_virt, nn.Linear):
                if add: # add eps
                    l_virt.weight = torch.nn.Parameter(l_orig.weight + l_eps.weight)
                    l_virt.bias = torch.nn.Parameter(l_orig.bias + l_eps.bias)
                else:   # subtract eps
                    l_virt.weight = torch.nn.Parameter(l_orig.weight - l_eps.weight)
                    l_virt.bias = torch.nn.Parameter(l_orig.bias - l_eps.bias)


    def subtract_noise_from_virtual_env(self):
        self.add_noise_to_virtual_env(add=False)


    def invert_eps(self):
        for l_eps in self.eps.modules():
            if isinstance(l_eps, nn.Linear):
                l_eps.weight = torch.nn.Parameter(-l_eps.weight)
                l_eps.bias = torch.nn.Parameter(-l_eps.bias)


    def calc_kl_div(self, counter, num_states):
        kl_div = 0

        csm = sum(counter.values())
        percs = [count/csm for count in counter.values()]

        for perc in percs:
            kl_div += perc * math.log(perc*num_states) # = perc * math.log(perc / (1/num_states))

        return kl_div


    def test_agent_on_real_env(self, agent, time_remaining, gtn_iteration):
        env = self.env_factory.generate_default_real_env('Test: ')

        t_s = time.time()

        all_states = []
        reward_list = []
        correct_path_perc = []
        test_episodes = agent.test_episodes

        rb_all = self.create_replay_buffer()

        agent.test_episodes = 1
        for i in range(test_episodes):
            reward, replay_buffer = agent.test(env=env,
                                               time_remaining=time_remaining - (time.time()-t_s),
                                               gtn_iteration=gtn_iteration)
            reward_list.append(reward[0])
            states, actions, next_states, rewards, dones = replay_buffer.get_all()
            next_states = [next_state.item() for next_state in next_states.int()]
            actions = [action.item() for action in actions.int()]

            all_states += next_states

            o_last = 0
            o_max = len(self.optimal_path)
            # FIXME: ignores last state
            for state in states:
                if state == self.optimal_path[o_last]:
                    o_last += 1
                if o_last == o_max - 1:
                    break

            correct_path_perc.append(o_last/o_max)
            rb_all.merge_buffer(replay_buffer)

        agent.test_episodes = test_episodes

        #print(agent.test_counter / agent.total_counter)
        #print(self.optimal_path)
        #different_state_perc = len(set(all_states)) / env.get_state_dim()

        state_counter = Counter(all_states)
        kl_div = self.calc_kl_div(state_counter, env.get_state_dim())
        correct_path_perc = statistics.mean(correct_path_perc)
        #print('{:3f} {:3f}'.format(different_state_perc, correct_path_perc))
        #print(correct_path_perc)

        #print(set(all_states))
        #print(agent.q_table)
        #states_visited_during_learning = [int(abs(sum(q_vals))>1e-5) for q_vals in agent.q_table]
        #print(states_visited_during_learning)
        #print(agent.q_table)
        #print(correct_path_perc)
        #print(different_state_perc)
        #print(kl_div)

        mean_reward = sum(reward_list) / len(reward_list) \
                      - kl_div * self.exploration_gain \
                      + correct_path_perc * self.correct_path_gain#+ different_state_perc*0.01 #+ correct_path_perc*0.3
        #mean_reward = sum(reward_list) / len(reward_list) - kl_div * 0.3 #+ correct_path_perc * 0.3
        return mean_reward, rb_all


    # def test_agent_on_real_env(self, agent, time_remaining, gtn_iteration):
    #     env = self.env_factory.generate_default_real_env('Test: ')
    #     reward_list, replay_buffer = agent.test(env=env,
    #                                             time_remaining=time_remaining,
    #                                             gtn_iteration=gtn_iteration)
    #
    #     different_states = len(replay_buffer.get_all()[0].int().unique()) / env.get_state_dim()
    #     #print(different_states)
    #
    #     mean_reward = sum(reward_list) / len(reward_list) + different_states*0.3
    #     return mean_reward


def run_gtn_on_single_pc(config):
    def run_gtn_worker(config, id):
        gtn = GTN_Worker(config, id)
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

