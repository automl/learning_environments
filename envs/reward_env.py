import torch
import torch.nn as nn
from models.model_utils import build_nn_from_config
from utils import from_one_hot_encoding, to_one_hot_encoding

class RewardEnv(nn.Module):
    def __init__(self, real_env, kwargs):
        super().__init__()

        self.env_name = str(kwargs["env_name"])
        self.device = str(kwargs["device"])
        self.state_dim = int(kwargs["state_dim"])
        self.action_dim = int(kwargs["action_dim"])
        self.solved_reward = int(kwargs["solved_reward"])
        self.reward_env_type = int(kwargs["reward_env_type"])

        # for gym compatibility
        self._max_episode_steps = int(kwargs["max_steps"])
        self.action_space = kwargs["action_space"]
        self.observation_space = kwargs["observation_space"]

        # initialize two environments
        self.real_env = real_env
        self.reward_env = self.build_reward_net(kwargs)
        self.state = self.reset()


    def build_reward_net(self, kwargs):
        if self.reward_env_type == 0:
            return None
        if self.reward_env_type == 1 or self.reward_env_type == 2:   # potential based reward function
            input_dim = 2*self.state_dim + self.action_dim
        elif self.reward_env_type == 3 or self.reward_env_type == 4:
            input_dim = self.state_dim
        else:
            raise NotImplementedError('Unknown reward_env_type: ' + str(self.reward_env_type))

        return build_nn_from_config(input_dim=input_dim,
                                    output_dim=1,
                                    nn_config=kwargs).to(self.device)

    def step(self, action):
        next_state, reward, done, _ = self.real_env.step(action)

        reward = self._calc_reward(state=self.state,
                                   action=action,
                                   next_state=next_state,
                                   reward=reward)
        self.state = next_state

        return next_state, reward, done, {}

    def _calc_reward(self, state, action, next_state, reward):
        state_torch = torch.tensor(state, device=self.device, dtype=torch.float32)
        action_torch = torch.tensor(action, device=self.device, dtype=torch.float32)
        next_state_torch = torch.tensor(next_state, device=self.device, dtype=torch.float32)
        reward_torch = torch.tensor(reward, device=self.device, dtype=torch.float32)

        if self.reward_env_type == 0:
            reward_res = reward_torch

        if self.reward_env_type == 1:
            input = torch.cat((state_torch, action_torch, next_state_torch), dim=action_torch.dim()-1)
            reward_res = self.reward_env(input)

        elif self.reward_env_type == 2:
            input = torch.cat((state_torch, action_torch, next_state_torch), dim=action_torch.dim()-1)
            reward_res = reward_torch + self.reward_env(input)

        elif self.reward_env_type == 3:
            reward_res = self.reward_env(state_torch)

        elif self.reward_env_type == 4:
            reward_add = self.reward_env(state_torch)
            reward_res = reward_torch + reward_add

        return reward_res.item()


    def seed(self, seed):
        return self.real_env.seed(seed)

    def render(self):
        return self.real_env.render()

    def reset(self):
        self.state = self.real_env.reset()
        return self.state

    def close(self):
        return self.real_env.close()

