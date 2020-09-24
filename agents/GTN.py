import yaml
import random
import torch
import torch.nn as nn
import numpy as np
import os
import copy
import higher
from time import time
from agents.TD3 import TD3
from agents.agent_utils import select_agent
from agents.env_matcher import EnvMatcher
from agents.REPTILE import reptile_train_agent
from envs.env_factory import EnvFactory
from utils import print_abs_param_sum

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GTN(nn.Module):
    def __init__(self, config):
        super().__init__()
        # for saving/loading
        self.config = config

        gtn_config = config["agents"]["gtn"]
        self.max_iterations = gtn_config["max_iterations"]
        self.real_step_size = gtn_config["real_step_size"]
        self.virtual_step_size = gtn_config["virtual_step_size"]
        self.pretrain_env = gtn_config["pretrain_env"]
        self.pretrain_agent = gtn_config["pretrain_agent"]
        self.type = []
        for i in range(10):
            strng = "type_" + str(i)
            self.type.append(gtn_config[strng])

        self.agent_name = gtn_config["agent_name"]
        self.agent = select_agent(config, self.agent_name)
        self.env_matcher = EnvMatcher(config)

        self.env_factory = EnvFactory(config)
        self.virtual_env = self.env_factory.generate_default_virtual_env()
        self.real_env = self.env_factory.generate_default_real_env()

        different_envs = gtn_config["different_envs"]
        self.input_seeds = [self.env_factory.generate_default_input_seed()]
        # generate multiple different real envs with associated seed
        for i in range(different_envs-1):
            self.input_seeds.append(self.env_factory.generate_random_input_seed())

    def print_stats(self):
        print_abs_param_sum(self.virtual_env, "VirtualEnv")
        print_abs_param_sum(self.agent.actor, "Actor")
        print_abs_param_sum(self.agent.critic_1, "Critic1")
        print_abs_param_sum(self.agent.critic_2, "Critic2")

    def train(self):
        order = []
        timings = []

        self.print_stats()

        # t = time()
        # print("-- pretraining agent on real env --")
        # _, replay_buffer = reptile_train_agent(agent=self.agent,
        #                     env=self.real_env,
        #                     step_size=self.real_step_size)
        # timings.append(int(time() - t))
        #
        # t = time()
        # print("-- matching virtual env to real env --")
        # self.env_matcher.train(virtual_env=self.virtual_env,
        #                        input_seeds=self.input_seeds,
        #                        replay_buffer=replay_buffer)
        # timings.append(int(time()-t))

        for it in range(self.max_iterations):
            self.print_stats()
            t = time()

            print("-- training on virtual env --")
            self.agent.train(env=self.virtual_env)

            with higher.innerloop_ctx(self.agent, self.agent.actor_optimizer) as (fagent, diffopt):

                print("-- evaluate on real env and update virtual env--")
                reward_list, _ = fagent.train(env=self.real_env, diffopt=diffopt)

                loss = -reward_list

                adam_params = list(self.virtual_env.parameters())
                self.virtual_env_optimizer = torch.optim.Adam(adam_params, lr=self.lr)
                self.virtual_env.requires_grad = True
                self.virtual_env_optimizer.zero_grad()

                loss.backward()





            timings.append(int(time()-t))

        self.print_stats()

        return timings

    def test(self):
        # generate 10 different deterministic environments with increasing difficulty
        # and check for every environment how many episodes it takes the agent to solve it
        # N.B. we have to reset the state of the agent before every iteration

        # todo future: fine-tuning, then test
        # to avoid problems with wrongly initialized optimizers
        if isinstance(self.agent, TD3):
            env = self.env_factory.generate_default_real_env()
            self.agent.reset_optimizer()

        mean_episodes_till_solved = 0
        episodes_till_solved = []
        agent_state = copy.deepcopy(self.agent.get_state_dict())

        if self.config['env_name'] == 'HalfCheetah-v2':
            interpolate_vals = [0, 0.02, 0.1, 0.4, 1]
        else:
            interpolate_vals = np.arange(0, 1.01, 0.2)

        for interpolate in interpolate_vals:
            self.agent.set_state_dict(agent_state)
            self.print_stats()
            env = self.env_factory.generate_interpolated_real_env(interpolate)
            reward_list, replay_buffer = self.agent.train(env=env)
            mean_episodes_till_solved += len(reward_list)
            episodes_till_solved.append(len(reward_list))
            print("episodes till solved: " + str(len(reward_list)))

        self.agent.set_state_dict(agent_state)
        mean_episodes_till_solved /= len(interpolate_vals)

        return mean_episodes_till_solved, episodes_till_solved

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


if __name__ == "__main__":
    with open("../default_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    #config = {'env_name': 'MountainCarContinuous-v0', 'seed': 0, 'render_env': True, 'agents': {'gtn': {'max_iterations': 5, 'type_0': 1, 'type_1': 3, 'type_2': 1, 'type_3': 2, 'type_4': 2, 'type_5': 1, 'type_6': 1, 'type_7': 1, 'type_8': 1, 'type_9': 1, 'different_envs': 2, 'agent_name': 'TD3', 'match_step_size': 0.11390246115390855, 'real_step_size': 0.1359401227445613, 'virtual_step_size': 0.06912544738161745, 'both_step_size': 0.5404661477307044, 'pretrain_env': True, 'pretrain_agent': False}, 'reptile': {'max_iterations': 10000, 'step_size': 0.1, 'agent_name': 'TD3'}, 'env_matcher': {'oversampling': 2.06100060173854, 'lr': 0.0008712488562961471, 'weight_decay': 0.0, 'batch_size': 256, 'early_out_diff': 0.0, 'early_out_num': 50, 'max_steps': 7, 'step_size': 146, 'gamma': 0.5, 'use_rb': True}, 'td3': {'max_episodes': 100, 'init_episodes': 50, 'batch_size': 256, 'gamma': 0.9981358228002741, 'lr': 0.0011235253745535201, 'weight_decay': 0.0, 'tau': 0.033415629036680126, 'policy_delay': 2, 'rb_size': 1000000, 'same_action_num': 1, 'activation_fn': 'relu', 'hidden_size': 246, 'hidden_layer': 1, 'weight_norm': False, 'action_std': 0.22859425627759783, 'early_out_num': 5, 'optim_env_with_actor': False, 'optim_env_with_critic': True, 'match_weight_actor': 0.010667452876528967, 'match_weight_critic': 1.547971125672462, 'match_batch_size': 128, 'match_oversampling': 1.2555463378821679, 'match_delay': 4, 'virtual_min_episodes': 4, 'both_min_episodes': 3}, 'ppo': {'max_episodes': 100000, 'update_episodes': 20.0, 'ppo_epochs': 100, 'gamma': 0.99, 'lr': 0.001, 'weight_decay': 0.0, 'vf_coef': 0.5, 'ent_coef': 0.01, 'eps_clip': 0.2, 'same_action_num': 1, 'activation_fn': 'leakyrelu', 'hidden_size': 128, 'hidden_layer': 1, 'weight_norm': False, 'action_std': 0.5, 'early_out_num': 50}}, 'envs': {'Pendulum-v0': {'solved_reward': -300.0, 'max_steps': 200, 'dt': 0.05, 'max_speed': [8, 10, 12, False], 'max_torque': [1.5, 2, 3, False], 'g': [5, 10, 15, True], 'm': [0.5, 1, 1.2, True], 'l': [0.5, 1, 1.2, True], 'activation_fn': 'leakyrelu', 'hidden_size': 256, 'hidden_layer': 2, 'input_seed_dim': 10, 'input_seed_mean': 0.1, 'input_seed_range': 0.1, 'zero_init': False, 'weight_norm': True}, 'MountainCarContinuous-v0': {'solved_reward': 50.0, 'max_steps': 999, 'max_speed': [0.05, 0.07, 0.1, False], 'power': [0.0012, 0.0015, 0.002, False], 'goal_position': [0.45, 0.45, 0.5, True], 'goal_velocity': [0.0, 0.0, 0.05, True], 'activation_fn': 'leakyrelu', 'hidden_size': 66, 'hidden_layer': 2, 'input_seed_dim': 8, 'input_seed_mean': 0.09327506740580452, 'input_seed_range': 0.03496039777502475, 'zero_init': False, 'weight_norm': True}, 'Test': {'solved_reward': 50.0, 'max_steps': 200, 'activation_fn': 'leakyrelu', 'hidden_size': 128, 'hidden_layer': 1, 'input_seed_dim': 3, 'input_seed_mean': 0.1, 'input_seed_range': 0.1, 'zero_init': False, 'weight_norm': True}, 'HalfCheetah-v2': {'solved_reward': 3000.0, 'max_steps': 1000, 'g': [-1, -9.806, -18, True], 'cripple_joint': False, 'activation_fn': 'tanh', 'hidden_size': 256, 'hidden_layer': 2, 'input_seed_dim': 3, 'input_seed_mean': 0.1, 'input_seed_range': 0.1, 'zero_init': False, 'weight_norm': True}}}
    # set seeds
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    gtn = GTN(config)
    gtn.train()
    result = gtn.test()
    print(result)
