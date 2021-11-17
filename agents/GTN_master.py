import copy

import torch
import torch.nn as nn
import os
import time
import numpy as np
import random
import string
import statistics
from datetime import datetime
from agents.GTN_base import GTN_Base
from envs.env_factory import EnvFactory
from agents.utils import calc_abs_param_sum
from communicate.helpers_communication import x_minutes_passed
from communicate.tcp_master_selector import connections_for_later, lost_connections


def make_list_of_all_workers_available(ids_in_master: dict):
    for key in lost_connections.keys():
        ids_in_master.pop(key)


def populate_ids():
    return copy.deepcopy(connections_for_later)


def get_ids(id_dict: dict):
    list = [val["id"] for k, val in id_dict.items()]
    return list


class GTN_Master(GTN_Base):
    def __init__(self, config, bohb_id=-1, bohb_working_dir=None, additional_arguments=None):
        super().__init__(bohb_id)
        self.debug = False
        self.config = config
        self.device = config["device"]
        self.env_name = config['env_name']

        gtn_config = config["agents"]["gtn"]
        self.max_iterations = gtn_config["max_iterations"]
        self.agent_name = gtn_config["agent_name"]
        self.num_workers = gtn_config["num_workers"] + 1
        self.step_size = gtn_config["step_size"]
        self.nes_step_size = gtn_config["nes_step_size"]
        self.weight_decay = gtn_config["weight_decay"]
        self.score_transform_type = gtn_config["score_transform_type"]
        self.time_mult = gtn_config["time_mult"]
        self.time_max = gtn_config["time_max"]  # default is roughly 30 minutes
        self.time_max = self.time_max
        if self.debug:
            self.time_max = self.time_max / 5
        self.time_sleep_master = gtn_config["time_sleep_master"]
        self.quit_when_solved = gtn_config["quit_when_solved"]
        self.synthetic_env_type = gtn_config["synthetic_env_type"]
        self.unsolved_weight = gtn_config["unsolved_weight"]

        # make it faster on single PC
        if gtn_config["mode"] == 'single':
            self.time_sleep_master /= 10

        # to store results from workers
        self.all_time_elapsed_list = [self.time_max] * self.num_workers  # for debugging
        self.all_score_list = [0] * self.num_workers
        self.all_score_orig_list = [0] * self.num_workers  # for debugging
        self.all_score_transform_list = [0] * self.num_workers

        # to keep track of the reference virtual env
        self.env_factory = EnvFactory(config)
        if self.synthetic_env_type == 0:
            generate_synthetic_env_fn = self.env_factory.generate_virtual_env
        elif self.synthetic_env_type == 1:
            generate_synthetic_env_fn = self.env_factory.generate_reward_env
        else:
            raise NotImplementedError("Unknown synthetic_env_type value: " + str(self.synthetic_env_type))

        self.synthetic_env_orig = generate_synthetic_env_fn(print_str='GTN_Base: ')
        self.synthetic_env_list = [generate_synthetic_env_fn(print_str='GTN_Master: ') for _ in range(self.num_workers)]
        self.eps_list = [generate_synthetic_env_fn(print_str='GTN_Master: ') for _ in range(self.num_workers)]

        # for early out
        self.avg_runtime = None
        self.real_env = self.env_factory.generate_real_env()

        # to store models
        if bohb_working_dir:
            self.model_dir = str(os.path.join(bohb_working_dir, 'GTN_models_' + self.env_name))
        else:
            self.model_dir = str(os.path.join(os.getcwd(), "results", 'GTN_models_' + self.env_name))
        self.model_name = self.get_model_file_name(self.env_name + '_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=6)) + '.pt')
        self.best_score = -float('Inf')

        os.makedirs(self.model_dir, exist_ok=True)

        print('Starting GTN Master with bohb_id {}'.format(bohb_id))

        # self.bohb_next_run_counter = 0

        self.additional_arguments = additional_arguments  # Currently not used, added for later

        # For Communication Purposes
        self.started_at_time = datetime.now()
        self.minutes_till_reading = 30
        if self.debug:
            self.minutes_till_reading = 1
        self.available_workers = None
        self.active_ids = []
        self.iteration_counter = 0

    def run(self):
        mean_score_orig_list = []

        for it in range(self.max_iterations):
            self.available_workers = copy.deepcopy(connections_for_later)
            t1 = time.time()
            print('-- Master: Iteration ' + str(it) + ' ' + str(time.time() - t1))
            print('-- Master: write worker inputs' + ' ' + str(time.time() - t1))
            self.write_worker_inputs(it)

            while not x_minutes_passed(start=self.started_at_time, end=datetime.now(), minutes_passed=self.minutes_till_reading):
                # wait for some minutes till wokers to come up, or all workers are up -> start reading
                time.sleep(self.time_sleep_master)
                if len(connections_for_later) == self.num_workers:
                    break

            # X minutes passsed -> we make a list of available workers:
            self.available_workers = copy.deepcopy(connections_for_later)  # copy dict over to not have a thread read/write problem (connections_for_later is used in this main as well as the communication thread)
            print("self.available_ids: ", self.available_workers)

            print('-- Master: read worker results (tying)' + ' at elapsed time:' + str(time.time() - t1))
            self.read_worker_results()

            active_scores_array = np.array(self.all_score_list)[self.active_ids[-1]]  # only calculate the mean score for workers that were active
            mean_score = np.mean(active_scores_array)
            mean_score_orig_list.append(mean_score)
            solved_flag = self.save_good_model(mean_score)

            if solved_flag and self.quit_when_solved:
                print('ENV SOLVED')
                # self.bohb_next_run_counter += 1
                break

            print('-- Master: rank transform' + ' ' + str(time.time() - t1))
            self.score_transform()
            print('-- Master: update env' + ' ' + str(time.time() - t1))
            self.update_env()
            print('-- Master: print statistics' + ' ' + str(time.time() - t1))
            self.print_statistics(it=it, time_elapsed=time.time() - t1)

            self.iteration_counter += 1

        print('Master quitting')

        self.print_statistics(it=-1, time_elapsed=-1)

        # error handling
        if len(mean_score_orig_list) > 0:
            return np.mean(self.score_orig_list), mean_score_orig_list, self.model_name
        else:
            return 1e9, mean_score_orig_list, self.model_name

    def write_worker_inputs(self, it):
        # timeout = self.calc_worker_timeout() # old. lead to sometimes no execution in worker
        timeout = self.time_max  # new
        print('timeout: ' + str(timeout))

        if it == 0:
            id_list = [i + 1 for i in range(self.num_workers)]
        else:
            id_list = get_ids(self.available_workers)

        print(f"Iteration: {it}, wrinting files for ids: {id_list}")
        for id in id_list:

            file_name = self.get_input_file_name(id=id)
            check_file_name = self.get_input_check_file_name(id=id)

            time.sleep(self.time_sleep_master)

            # if we are not using bohb, shut everything down after last iteration
            if self.bohb_id < 0:
                quit_flag = it == self.max_iterations - 1
            else:
                quit_flag = False

            data = {}
            data['timeout'] = timeout
            data['quit_flag'] = quit_flag
            data['config'] = self.config
            data['synthetic_env_orig'] = self.synthetic_env_orig.state_dict()
            # data['bohb_next_run_counter'] = self.bohb_next_run_counter

            torch.save(data, file_name)
            torch.save({}, check_file_name)

    def read_worker_results(self):
        checked_this_iteration = []
        delete_these_files = []
        while len(self.available_workers) > 0:
            self.update_ids_to_check()  # check if a worker aborted since last check and adapt lists
            for id in get_ids(self.available_workers):
                if id in checked_this_iteration:
                    continue
                file_name = self.get_result_file_name(id)
                check_file_name = self.get_result_check_file_name(id)

                # wait until worker has finished calculations
                if not os.path.isfile(check_file_name):
                    continue
                else:
                    print(f"found: {check_file_name}")
                    self.remove_id(id)
                    checked_this_iteration.append(id)
                    delete_these_files.append(check_file_name)
                    delete_these_files.append(file_name)

                    data = torch.load(file_name)
                    self.all_time_elapsed_list[id ] = data['time_elapsed']
                    self.all_score_list[id ] = data['score']
                    self.eps_list[id ].load_state_dict(data['eps'])
                    self.all_score_orig_list[id ] = data['score_orig']
                    self.synthetic_env_list[id ].load_state_dict(data['synthetic_env'])  # for debugging

        self.active_ids.append(checked_this_iteration)  # keep log of active ids for score transformation

        for file_name in delete_these_files:
            os.remove(file_name)
            print(f"deleting: {file_name}")

    def save_good_model(self, mean_score):
        if self.synthetic_env_orig.is_virtual_env():
            print("DEBUG IF")
            if mean_score > self.real_env.get_solved_reward() and mean_score > self.best_score:
                self.save_model()
                self.best_score = mean_score
                return True
        else:
            # we save all models and select the best from the log
            # whether we can solve an environment is irrelevant for reward_env since we optimize for speed here
            if mean_score > self.best_score:
                self.save_model()
                self.best_score = mean_score

        save_dict = {}
        save_dict['model'] = self.synthetic_env_orig.state_dict()
        save_dict['config'] = self.config
        scores = np.asarray(self.all_score_list)[self.active_ids[-1]]
        save_dict['scores'] = list(scores)
        save_path = os.path.join(self.model_dir, f"iter_{self.iteration_counter}.pt")
        print('saving not solved model: ' + str(save_path))
        torch.save(save_dict, save_path)

        return False

    def save_model(self):
        save_dict = {}
        save_dict['model'] = self.synthetic_env_orig.state_dict()
        save_dict['config'] = self.config
        save_path = os.path.join(self.model_dir, self.model_name)
        print('save model: ' + str(save_path))
        torch.save(save_dict, save_path)

    def get_model_file_name(self, file_name):
        return os.path.join(self.model_dir, file_name)

    def remove_id(self, id):
        key_to_delete = None
        for k, v in self.available_workers.items():
            if v["id"] == id:
                key_to_delete = k
        assert key_to_delete is not None, "key_to_delete was None"
        self.available_workers.pop(key_to_delete)

    def update_ids_to_check(self):
        for k, v in lost_connections.items():
            if k in self.available_workers:
                self.available_workers.pop(k)
                print("self.available_ids: ", self.available_workers)

    def calc_worker_timeout(self):
        if self.all_time_elapsed_list[0] is None:
            return self.time_max
        else:
            return statistics.mean(self.all_time_elapsed_list) * self.time_mult

    def score_transform(self):
        scores = np.asarray(self.all_score_list)[self.active_ids[-1]]  # only select scores from workers that were active
        scores_orig = np.asarray(self.all_score_orig_list)[self.active_ids[-1]]  # only select scores from workers that were active

        if self.score_transform_type == 0:
            # convert [1, 0, 5] to [0.2, 0, 1]
            scores = (scores - min(scores)) / (max(scores) - min(scores) + 1e-9)

        elif self.score_transform_type == 1:
            # convert [1, 0, 5] to [0.5, 0, 1]
            s = np.argsort(scores)
            n = len(scores)
            for i in range(n):
                scores[s[i]] = i / (n - 1)

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
            # consider single best eps that is better than the average
            avg_score_orig = np.mean(scores_orig)

            scores_idx = np.where(scores > avg_score_orig + 1e-6, 1, 0)  # 1e-6 to counter numerical errors
            if sum(scores_idx) > 0:
                scores_tmp = np.zeros(scores.size)
                scores_tmp[np.argmax(scores)] = 1
                scores = scores_tmp
            else:
                scores = scores_idx

        elif self.score_transform_type == 6 or self.score_transform_type == 7:
            # consider all eps that are better than the average, normalize weight sum to 1
            avg_score_orig = np.mean(scores_orig)

            scores_idx = np.where(scores > avg_score_orig + 1e-6, 1, 0)  # 1e-6 to counter numerical errors
            if sum(scores_idx) > 0:
                # if sum(scores_idx) > 0:
                scores = scores_idx * (scores - avg_score_orig) / (max(scores) - avg_score_orig + 1e-9)
                if self.score_transform_type == 6:
                    scores /= max(scores)
                else:
                    scores /= sum(scores)
            else:
                scores = scores_idx

        else:
            raise ValueError("Unknown rank transform type: " + str(self.score_transform_type))

        # copy values (for active workers) over into list
        act_idxs = self.active_ids[-1]
        new_score_array = np.copy(self.all_score_transform_list)
        for i, id in enumerate(act_idxs):
            new_score_array[id] = scores[i]

        self.all_score_transform_list = new_score_array.tolist()

    def update_env(self):
        ss = self.step_size

        if self.nes_step_size:
            ss = ss / self.num_workers

        # print('-- update env --')
        print('score_orig_list      ' + str(np.array(self.all_score_orig_list)[self.active_ids[-1]]))
        print('score_list           ' + str(np.array(self.all_score_list)[self.active_ids[-1]]))
        print('score_transform_list ' + str(np.array(self.all_score_transform_list)[self.active_ids[-1]]))
        print('venv weights         ' + str([calc_abs_param_sum(elem).item() for elem in self.synthetic_env_list]))

        print('weights before: ' + str(calc_abs_param_sum(self.synthetic_env_orig).item()))

        # weight decay
        for l_orig in self.synthetic_env_orig.modules():
            if isinstance(l_orig, nn.Linear):
                l_orig.weight = torch.nn.Parameter(l_orig.weight * (1 - self.weight_decay))
                if l_orig.bias != None:
                    l_orig.bias = torch.nn.Parameter(l_orig.bias * (1 - self.weight_decay))

        print('weights after weight decay: ' + str(calc_abs_param_sum(self.synthetic_env_orig).item()))

        # Only use active workers:
        active_eps = list(np.array(self.eps_list)[self.active_ids[-1]])
        active_scores = list(np.array(self.all_score_transform_list)[self.active_ids[-1]])

        # weight update
        for eps, score_transform in zip(active_eps, active_scores):
            for l_orig, l_eps in zip(self.synthetic_env_orig.modules(), eps.modules()):
                if isinstance(l_orig, nn.Linear):
                    l_orig.weight = torch.nn.Parameter(l_orig.weight + ss * score_transform * l_eps.weight)
                    if l_orig.bias != None:
                        l_orig.bias = torch.nn.Parameter(l_orig.bias + ss * score_transform * l_eps.bias)

        print('weights after update: ' + str(calc_abs_param_sum(self.synthetic_env_orig).item()))

    def print_statistics(self, it, time_elapsed):
        orig_score = statistics.mean(self.all_score_orig_list)
        mean_time_elapsed = statistics.mean(self.all_time_elapsed_list)
        print('--------------')
        print('GTN iteration:    ' + str(it))
        print('GTN available workers:    ' + str(len(self.active_ids)) + ' : ' + str(self.active_ids[-1]))
        print('GTN mstr t_elaps: ' + str(time_elapsed))
        print('GTN avg wo t_elaps: ' + str(mean_time_elapsed))
        print('GTN avg eval score:   ' + str(orig_score))
        print('--------------')
