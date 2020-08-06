import torch
import torch.nn as nn
import numpy as np
from models.model_utils import build_nn_from_config
from envs.env_utils import generate_env_with_kwargs
import gym.envs.mujoco.mujoco_env
import gym.envs.mujoco.half_cheetah


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VirtualEnv(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.input_seed = None
        self.env_name = str(kwargs["env_name"])
        self.state_dim = int(kwargs["state_dim"])
        self.action_dim = int(kwargs["action_dim"])
        self.input_seed_dim = int(kwargs["input_seed_dim"])
        self.zero_init = bool(kwargs["zero_init"])
        self.solved_reward = float(kwargs["solved_reward"])
        # todo fabio: maybe change to random init state
        self.state = torch.zeros(self.state_dim, device=device)  # not sure what the default state should be

        # for gym compatibility
        self._max_episode_steps = int(kwargs["max_steps"])

        # instantiate own small env for rendering
        self.gym_env = generate_env_with_kwargs(kwargs, self.env_name)
        self.gym_env.reset()

        self.reset_flag = False

        self.state_net = build_nn_from_config(input_dim=self.state_dim + self.action_dim + self.input_seed_dim,
                                              output_dim=self.state_dim,
                                              nn_config=kwargs).to(device)
        self.reward_net = build_nn_from_config(input_dim=self.state_dim + self.action_dim + self.input_seed_dim,
                                               output_dim=1,
                                               nn_config=kwargs).to(device)
        self.done_net = build_nn_from_config(input_dim=self.state_dim + self.action_dim + self.input_seed_dim,
                                             output_dim=1,
                                             nn_config=kwargs).to(device)


    def reset(self):
        self.reset_flag = True
        if self.zero_init:
            return torch.zeros(self.state_dim, device=device)
        else:
            return torch.tensor(self.gym_env.reset(), device=device, dtype=torch.float32)


    def step(self, action, state, input_seed):
        self.reset_flag = False
        input = torch.cat((action, state, input_seed), dim=len(action.shape) - 1)
        next_state = self.state_net(input)
        reward = self.reward_net(input)
        done = self.done_net(input)
        return next_state, reward, done

    def render(self, state, action):
        try:
            if self.env_name == "HalfCheetah-v2":
                if hasattr(self, 'half_cheetah_x') and self.reset_flag is False:
                    x_pos = self.half_cheetah_x
                else:
                    x_pos = 0

                np_state = state.cpu().detach().numpy()
                self.gym_env.data.qpos[0] = x_pos
                self.gym_env.data.qpos[1:9] = np_state[:8]
                self.gym_env.do_simulation(action.cpu().detach().numpy(), 1)
                self.gym_env.render()

                self.half_cheetah_x = self.gym_env.data.qpos[0]
        except:
            pass

    def close(self):
        if self.env_name != "Test":
            # if self.gym_env.viewer:
            #     self.gym_env.viewer.close()
            #     self.gym_env.viewer = None
            if self.gym_env:
                self.gym_env.close()

    def get_state_dict(self):
        env_state = {}
        env_state["virtual_env_state_net"] = self.state_net.state_dict()
        env_state["virtual_env_reward_net"] = self.reward_net.state_dict()
        env_state["virtual_env_done_net"] = self.done_net.state_dict()
        return env_state

    def set_state_dict(self, env_state):
        self.state_net.load_state_dict(env_state["virtual_env_state_net"])
        self.reward_net.load_state_dict(env_state["virtual_env_reward_net"])
        self.done_net.load_state_dict(env_state["virtual_env_done_net"])
