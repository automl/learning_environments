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

        reptile_config = config["agents"]["reptile"]
        self.max_iterations = reptile_config["max_iterations"]
        self.step_size = reptile_config["step_size"]
        self.agent_name = reptile_config["agent_name"]
        self.export_path = config["export_path"]
        self.config = config

        self.env_factory = EnvFactory(config)
        self.agent = select_agent(config, self.agent_name)
        self.input_seeds = torch.tensor(
            [np.random.random() for _ in range(self.max_iterations)], device="cpu", dtype=torch.float32
        ).unsqueeze(1)

        self.virtual_env = self.env_factory.generate_default_virtual_env()

        # if os.path.isfile(self.export_path):
        #     self.load_checkpoint()

    def run(self):
        for it in range(self.max_iterations):

            # if it % 10 == 0:
            #     self.save_checkpoint()

            # train on real env for a bit
            self.real_env = self.env_factory.generate_default_real_env()  # todo: random or default real env?

            # map virtual env to real env
            self.match_environment(real_env = self.real_env,
                                   virtual_env = self.virtual_env,
                                   input_seed = self.input_seeds[it])

            # now train on virtual env
            print("-- training on real env --")
            self.reptile_run(env = self.real_env)

            # now train on virtual env
            print("-- training on virtual env --")
            self.reptile_run(env = self.virtual_env, input_seed = self.input_seeds[it])


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
        #target.load_state_dict(new_state_dict)


    def match_environment(self, real_env, virtual_env, input_seed):
        optimizer = torch.optim.Adam(virtual_env.parameters(), lr=1e-4)
        avg_meter_loss = AverageMeter(buffer_size=50,
                                      update_rate=10,
                                      print_str='Average loss: ')
        avg_meter_diff = AverageMeter(buffer_size=50,
                                      update_rate=10,
                                      print_str='Average diff: ')


        # todo: value from config
        for _ in range(10000):
            states_list = []
            actions_list = []
            next_states_list = []

            real_env.reset()

            # todo: value from config
            for k in range(256):
                # todo: better write getter instead of accessing the member variables directly

                # # use episodes
                # actions_list.append(real_env.env.action_space.sample())
                # if len(next_states_list) > 0:
                #     states_list.append(next_states_list[-1])
                #     next_state, _, _ = real_env.step(
                #         action=torch.tensor(actions_list[-1], device=device, dtype=torch.float32),
                #         state=torch.tensor(next_states_list[-1], device=device, dtype=torch.float32))
                #     next_states_list.append(next_state)
                # else:
                #     states_list.append(real_env.env.observation_space.sample())
                #     next_state, _, _ = real_env.step(
                #         action = torch.tensor(actions_list[-1], device=device, dtype=torch.float32),
                #         state = torch.tensor(states_list[-1], device=device, dtype=torch.float32))
                #     next_states_list.append(next_state)

                # use random actions/states
                states_list.append(real_env.env.observation_space.sample())
                actions_list.append(real_env.env.action_space.sample())
                next_state, _, _ = real_env.step(
                    action=torch.tensor(actions_list[-1], device=device, dtype=torch.float32),
                    state=torch.tensor(states_list[-1], device=device, dtype=torch.float32))
                next_states_list.append(next_state)

                # print('-------')
                # print(states_list[-1])
                # print(actions_list[-1])
                # print(next_states_list[-1])

            states = torch.tensor(states_list, device=device, dtype=torch.float32)
            actions = torch.tensor(actions_list, device=device, dtype=torch.float32)
            next_states_real = torch.stack(next_states_list)

            input_seeds = torch.tensor([input_seed], device=device, dtype=torch.float32).repeat(len(states)).unsqueeze(1)
            next_states_virtual, _, _ = virtual_env.step(action=actions, state=states, input_seed=input_seeds)

            diff = abs(next_states_real.cpu()-next_states_virtual.cpu()).sum()


            optimizer.zero_grad()
            loss = F.mse_loss(next_states_real, next_states_virtual)
            loss.backward()
            optimizer.step()

            avg_meter_loss.update(loss)
            avg_meter_diff.update(diff)




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
