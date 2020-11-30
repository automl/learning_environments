import torch
import torch.nn as nn
import os
import time
import math
import statistics
from collections import Counter
from agents.GTN_base import GTN_Base
from envs.env_factory import EnvFactory
from agents.agent_utils import select_agent


class GTN_Worker(GTN_Base):
    def __init__(self, id, bohb_id=-1):
        super().__init__(bohb_id)

        torch.manual_seed(id+bohb_id+int(time.time()))
        torch.cuda.manual_seed_all(id+bohb_id+int(time.time()))

        # for identifying the different workers
        self.id = id

        # flag to stop worker
        self.quit_flag = False
        self.time_sleep_worker = 2
        self.timeout = None
        self.uuid = None

        print('Starting GTN Worker with bohb_id {} and id {}'.format(bohb_id, id))


    def late_init(self, config):
        gtn_config = config["agents"]["gtn"]
        self.noise_std = gtn_config["noise_std"]
        self.num_test_envs = gtn_config["num_test_envs"]
        self.num_grad_evals = gtn_config["num_grad_evals"]
        self.grad_eval_type = gtn_config["grad_eval_type"]
        self.mirrored_sampling = gtn_config["mirrored_sampling"]
        self.exploration_gain = gtn_config["exploration_gain"]
        self.time_sleep_worker = gtn_config["time_sleep_worker"]
        self.minimize_score = gtn_config["minimize_score"]
        self.agent_name = gtn_config["agent_name"]
        self.synthetic_env_type = gtn_config["synthetic_env_type"]

        self.env_factory = EnvFactory(config)
        if self.synthetic_env_type == 0:
            generate_synthetic_env_fn = self.env_factory.generate_reward_env
        elif self.synthetic_env_type == 1:
            generate_synthetic_env_fn = self.env_factory.generate_virtual_env
        else:
            raise NotImplementedError("Unknown synthetic_env_type value: " + str(self.synthetic_env_type))

        self.synthetic_env_orig = generate_synthetic_env_fn(print_str='GTN_Base: ')
        self.synthetic_env = generate_synthetic_env_fn(print_str='GTN_Worker' + str(id) + ': ')
        self.eps = generate_synthetic_env_fn('GTN_Worker' + str(id) + ': ')


    def run(self):
        # read data from master
        while not self.quit_flag:
            self.read_worker_input()

            if self.quit_flag:
                print('QUIT FLAG')
                return

            time_start = time.time()

            # for evaluation purpose
            agent_orig = select_agent(config=self.config,
                                      agent_name=self.agent_name)
            agent_orig.train(env=self.synthetic_env_orig, time_remaining=self.timeout-(time.time()-time_start))
            score_orig = self.test_agent_on_real_env(agent=agent_orig, time_remaining=self.timeout-(time.time()-time_start))

            self.get_random_eps()

            # first mirrored noise +N
            self.add_noise_to_synthetic_env()

            score_add = []
            for i in range(self.num_grad_evals):
                #print('add ' + str(i))
                agent_add = select_agent(config=self.config, agent_name=self.agent_name)
                agent_add.train(env=self.synthetic_env, time_remaining=self.timeout-(time.time()-time_start))
                score = self.test_agent_on_real_env(agent=agent_add, time_remaining=self.timeout-(time.time()-time_start))
                score_add.append(score)

            # # second mirrored noise -N
            self.subtract_noise_from_synthetic_env()

            score_sub = []
            for i in range(self.num_grad_evals):
                #print('sub ' + str(i))
                agent_sub = select_agent(config=self.config, agent_name=self.agent_name)
                agent_sub.train(env=self.synthetic_env, time_remaining=self.timeout-(time.time()-time_start))
                score = self.test_agent_on_real_env(agent=agent_sub, time_remaining=self.timeout-(time.time()-time_start))
                score_sub.append(score)

            score_best = self.calc_best_score(score_add=score_add, score_sub=score_sub)

            self.write_worker_result(score=score_best,
                                     score_orig=score_orig,
                                     time_elapsed = time.time()-time_start)

        print('Worker ' + str(self.id) + ' quitting')


    def read_worker_input(self):
        file_name = self.get_input_file_name(id=self.id)
        check_file_name = self.get_input_check_file_name(id=self.id)

        while not os.path.isfile(check_file_name):
            if os.path.isfile(self.get_quit_file_name()):
                print('Worker {} {}: Emergency quit'.format(self.bohb_id, self.id))
                self.quit_flag = True
                return
            time.sleep(self.time_sleep_worker)
        time.sleep(self.time_sleep_worker)

        data = torch.load(file_name)

        self.uuid = data['uuid']
        self.timeout = data['timeout']
        self.quit_flag = data['quit_flag']
        self.config = data['config']

        self.late_init(self.config)

        self.synthetic_env_orig.load_state_dict(data['synthetic_env_orig'])
        self.synthetic_env.load_state_dict(data['synthetic_env_orig'])

        os.remove(check_file_name)
        os.remove(file_name)


    def write_worker_result(self, score, score_orig, time_elapsed):
        file_name = self.get_result_file_name(id=self.id)
        check_file_name = self.get_result_check_file_name(id=self.id)

        # wait until master has deleted the file (i.e. acknowledged the previous result)
        while os.path.isfile(file_name):
            time.sleep(self.time_sleep_worker)

        data = {}
        data["eps"] = self.eps.state_dict()
        data["synthetic_env"] = self.synthetic_env.state_dict() # for debugging
        data["time_elapsed"] = time_elapsed
        data["score"] = score
        data["score_orig"] = score_orig
        data["uuid"] = self.uuid
        torch.save(data, file_name)
        torch.save({}, check_file_name)


    def get_random_eps(self):
        for l_virt, l_eps in zip(self.synthetic_env.modules(), self.eps.modules()):
            if isinstance(l_virt, nn.Linear):
                l_eps.weight = torch.nn.Parameter(torch.normal(mean=torch.zeros_like(l_virt.weight),
                                                               std=torch.ones_like(l_virt.weight)) * self.noise_std)
                l_eps.bias = torch.nn.Parameter(torch.normal(mean=torch.zeros_like(l_virt.bias),
                                                             std=torch.ones_like(l_virt.bias)) * self.noise_std)


    def add_noise_to_synthetic_env(self, add=True):
        for l_orig, l_virt, l_eps in zip(self.synthetic_env_orig.modules(), self.synthetic_env.modules(), self.eps.modules()):
            if isinstance(l_virt, nn.Linear):
                if add: # add eps
                    l_virt.weight = torch.nn.Parameter(l_orig.weight + l_eps.weight)
                    l_virt.bias = torch.nn.Parameter(l_orig.bias + l_eps.bias)
                else:   # subtract eps
                    l_virt.weight = torch.nn.Parameter(l_orig.weight - l_eps.weight)
                    l_virt.bias = torch.nn.Parameter(l_orig.bias - l_eps.bias)


    def subtract_noise_from_synthetic_env(self):
        self.add_noise_to_synthetic_env(add=False)


    def invert_eps(self):
        for l_eps in self.eps.modules():
            if isinstance(l_eps, nn.Linear):
                l_eps.weight = torch.nn.Parameter(-l_eps.weight)
                l_eps.bias = torch.nn.Parameter(-l_eps.bias)


    def calc_best_score(self, score_sub, score_add):
        if self.minimize_score:
            # print('worker ' + str(self.id) + ' ' + str(score_sub) + ' ' + str(score_add) + ' ' + str(weight_sub) + ' ' + str(weight_add))
            if self.grad_eval_type == 'mean':
                score_sub = statistics.mean(score_sub)
                score_add = statistics.mean(score_add)
            elif self.grad_eval_type == 'minmax':
                score_sub = max(score_sub)
                score_add = max(score_add)
            else:
                raise NotImplementedError('Unknown parameter for grad_eval_type: ' + str(self.grad_eval_type))

            if self.mirrored_sampling:
                score_best = min(score_add, score_sub)
                if score_sub < score_add:
                    self.invert_eps()
                else:
                    self.add_noise_to_synthetic_env()
            else:
                score_best = score_add
                self.add_noise_to_synthetic_env()

        else:
            if self.grad_eval_type == 'mean':
                score_sub = statistics.mean(score_sub)
                score_add = statistics.mean(score_add)
            elif self.grad_eval_type == 'minmax':
                score_sub = min(score_sub)
                score_add = min(score_add)
            else:
                raise NotImplementedError('Unknown parameter for grad_eval_type: ' + str(self.grad_eval_type))

            if self.mirrored_sampling:
                score_best = max(score_add, score_sub)

                if score_sub > score_add:
                    self.invert_eps()
                else:
                    self.add_noise_to_synthetic_env()
            else:
                score_best = score_add
                self.add_noise_to_synthetic_env()

        return score_best


    def calc_kl_div(self, counter, num_states):
        kl_div = 0

        csm = sum(counter.values())
        percs = [count/csm for count in counter.values()]

        for perc in percs:
            kl_div += perc * math.log(perc*num_states) # = perc * math.log(perc / (1/num_states))

        return kl_div


    def test_agent_on_real_env(self, agent, time_remaining):
        env = self.env_factory.generate_real_env('Test: ')
        t_s = time.time()

        reward_list, replay_buffer = agent.test(env=env,
                                                time_remaining=time_remaining - (time.time() - t_s))

        #print(reward_list)

        if env.has_discrete_state_space():
            all_states = []

            states, actions, next_states, rewards, dones = replay_buffer.get_all()
            next_states = [next_state.item() for next_state in next_states.int()]

            all_states += next_states

            state_counter = Counter(all_states)
            kl_div = self.calc_kl_div(state_counter, env.get_state_dim())
            mean_reward = sum(reward_list) / len(reward_list) - kl_div * self.exploration_gain
        else:
            mean_reward = sum(reward_list) / len(reward_list)

        return mean_reward