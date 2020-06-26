import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from agents.agent_utils import select_agent
from envs.env_factory import EnvFactory
from utils import AverageMeter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GTN(nn.Module):
    def __init__(self, config):
        super().__init__()

        gtn_config = config["agents"]["gtn"]
        self.max_iterations = gtn_config["max_iterations"]
        self.match_lr = gtn_config["match_lr"]
        self.match_batch_size = gtn_config["match_batch_size"]
        self.match_iterations = gtn_config["match_iterations"]
        self.real_iterations = gtn_config["real_iterations"]
        self.virtual_iterations = gtn_config["virtual_iterations"]
        self.step_size = gtn_config["step_size"]

        agent_name = gtn_config["agent_name"]
        self.agent = select_agent(config, agent_name)

        different_envs = gtn_config["different_envs"]
        self.env_factory = EnvFactory(config)
        self.virtual_env = self.env_factory.generate_default_virtual_env()
        self.real_envs = []
        self.input_seeds = []
        if different_envs == 0:
            # generate single default environment with fixed seed
            self.real_envs.append(self.env_factory.generate_default_real_env())
            self.input_seeds.append(1)
        else:
            # generate multiple different real envs with associated seed
            for i in range(different_envs):
                self.real_envs.append(self.env_factory.generate_random_real_env())
                self.input_seeds.append(np.random.random())

        # if os.path.isfile(self.export_path):
        #     self.load_checkpoint()


    def run(self):
        for it in range(self.max_iterations):

            # if it % 10 == 0:
            #     self.save_checkpoint()

            # map virtual env to real env
            print("-- matching virtual env to real env --")
            self.match_environment(virtual_env = self.virtual_env,
                                   real_env = self.real_envs[it],
                                   input_seed = self.input_seeds[it])

            # now train on virtual env
            print("-- training on real env --")
            for _ in range(self.real_iterations):
                env_id = np.random.randint(len(self.real_envs))
                self.reptile_run(env = self.real_envs[env_id])

            # now train on virtual env
            print("-- training on virtual env --")
            for _ in range(self.virtual_iterations):
                env_id = np.random.randint(len(self.real_envs))
                self.reptile_run(env = self.virtual_env,
                                 input_seed = self.input_seeds[env_id])


    def reptile_run(self, env, input_seed=0):
        old_state_dict_agent = copy.deepcopy(self.agent.state_dict())
        if env.is_virtual_env():
            old_state_dict_env = copy.deepcopy(self.virtual_env.state_dict())

        self.agent.run(env=env, input_seed=input_seed)

        self.reptile_update(target = self.agent, old_state_dict = old_state_dict_agent)
        if env.is_virtual_env():
            self.reptile_update(target = env, old_state_dict = old_state_dict_env)


    def reptile_update(self, target, old_state_dict):
        new_state_dict = target.state_dict()
        for key, value in new_state_dict.items():
            new_state_dict[key] = old_state_dict[key] + (new_state_dict[key] - old_state_dict[key]) * self.step_size
        #target.load_state_dict(new_state_dict) # not needed?


    def match_environment(self, virtual_env, real_env, input_seed):
        old_state_dict_env = copy.deepcopy(virtual_env.state_dict())

        optimizer = torch.optim.Adam(virtual_env.parameters(), lr=self.match_lr)
        avg_meter_loss = AverageMeter(buffer_size=50,
                                      update_rate=50,
                                      print_str='Average loss: ')
        avg_meter_diff = AverageMeter(buffer_size=50,
                                      update_rate=50,
                                      print_str='Average diff: ')

        for _ in range(self.match_iterations):
            states_list = []
            actions_list = []
            outputs_list = []

            real_env.reset()

            for k in range(self.match_batch_size):
                # todo: maybe write getter instead of accessing the member variables directly
                # run random state/actions transitions on the real env
                states_list.append(real_env.env.observation_space.sample())
                actions_list.append(real_env.env.action_space.sample())
                next_state, reward, done = real_env.step(
                    action=torch.tensor(actions_list[-1], device=device, dtype=torch.float32),
                    state=torch.tensor(states_list[-1], device=device, dtype=torch.float32))
                outputs_list.append(torch.cat((next_state, reward.unsqueeze(0), done.unsqueeze(0)), dim=0))

            # convert to torch
            states = torch.tensor(states_list, device=device, dtype=torch.float32)
            actions = torch.tensor(actions_list, device=device, dtype=torch.float32)
            outputs_real = torch.stack(outputs_list)

            # simulate the same state/action transitions on the virtual env
            input_seeds = torch.tensor([input_seed], device=device, dtype=torch.float32).repeat(len(states)).unsqueeze(1)
            next_states_virtual, rewards_virtual, dones_virtual = virtual_env.step(action=actions, state=states, input_seed=input_seeds)
            outputs_virtual = torch.cat([next_states_virtual, rewards_virtual, dones_virtual], dim=1)

            # match virtual env to real env
            optimizer.zero_grad()
            loss = F.mse_loss(outputs_real, outputs_virtual)
            loss.backward()
            optimizer.step()

            # logging
            avg_meter_loss.update(loss)
            avg_meter_diff.update(abs(outputs_real.cpu()-outputs_virtual.cpu()).sum())

        self.reptile_update(target = virtual_env, old_state_dict = old_state_dict_env)


    def validate(self):
        # calculate after how many steps with a new environment a certain score is achieved
        env = self.env_factory.generate_default_real_env()
        results = self.agent.run(env=env)
        return sum(results[-20:-1])


    def save_checkpoint(self):
        if self.agent_name == "PPO":
            state_optimizer = {"optimizer": self.agent.optimizer.state_dict()}
        elif self.agent_name == "TD3":
            state_optimizer = {
                "critic_optimizer": self.agent.critic_optimizer.state_dict(),
                "actor_optimizer": self.agent.actor_optimizer.state_dict(),
            }
        state = {
            "agent_state_dict": self.agent.state_dict(),
            "env_factory": self.env_factory,
            "virtual_env_state_dict": self.virtual_env.state_dict(),
            "seeds": self.seeds,
            "config": self.config,  # not loaded
        }

        state = {**state, **state_optimizer}
        torch.save(state, self.export_path)


    def load_checkpoint(self):
        if os.path.isfile(self.export_path):
            state = torch.load(self.export_path)
            self.agent.load_state_dict(state["agent_state_dict"])
            self.env_factory = state["env_factory"]
            self.virtual_env.load_state_dict(state["virtual_env_state_dict"])
            self.seeds = state["seeds"]

            if self.agent_name == "PPO":
                self.agent.optimizer.load_state_dict(state["optimizer"])
            elif self.agent_name == "TD3":
                self.agent.critic_optimizer.load_state_dict(state["critic_optimizer"])
                self.agent.actor_optimizer.load_state_dict(state["actor_optimizer"]),

            print("=> loaded checkpoint '{}'".format(self.export_path))
