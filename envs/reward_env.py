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
        self.info_dim = int(kwargs["info_dim"])
        self.solved_reward = float(kwargs["solved_reward"])
        self.reward_env_type = int(kwargs["reward_env_type"])

        # for gym compatibility
        self._max_episode_steps = int(kwargs["max_steps"])
        self.action_space = kwargs["action_space"]
        self.observation_space = kwargs["observation_space"]

        # initialize two environments
        self.real_env = real_env
        self.reward_net = self.build_reward_net(kwargs)
        self.state = self.reset()


    def build_reward_net(self, kwargs):
        # 0: original reward
        # 1: native potential function (exclusive)
        # 2: native potential function (additive)
        # 3: potential function with additional info vector (exclusive)
        # 4: potential function with additional info vector (additive)
        # 101: weighted info vector as baseline (exclusive)
        # 102: weighted info vector as baseline (additive)

        if self.reward_env_type < 100:
            if self.reward_env_type == 0:
                input_dim = 1   # dummy dimension
            elif self.reward_env_type == 1 or self.reward_env_type == 2:
                input_dim = self.state_dim
            elif self.reward_env_type == 3 or self.reward_env_type == 4:
                input_dim = self.state_dim + self.info_dim
            else:
                raise NotImplementedError('Unknown reward_env_type: ' + str(self.reward_env_type))

            return build_nn_from_config(input_dim=input_dim,
                                        output_dim=1,
                                        nn_config=kwargs).to(self.device)
        else:
            if self.reward_env_type == 101 or self.reward_env_type == 102:
                return torch.nn.Parameter(torch.zeros(self.info_dim, device=self.device))
            else:
                raise NotImplementedError('Unknown reward_env_type: ' + str(self.reward_env_type))


    def step(self, action):
        next_state, reward, done, info = self.real_env.step(action)
        reward_res = self._calc_reward(state=self.state, action=action, next_state=next_state, reward=reward, info=info)
        self.state = next_state
        return next_state, reward_res, done, {}


    def _calc_reward(self, state, action, next_state, reward, info):
        if 'TimeLimit.truncated' in info:   # remove additional information from wrapper
            info.pop('TimeLimit.truncated')

        action_torch = torch.tensor(action, device=self.device, dtype=torch.float32)
        reward_torch = torch.tensor(reward, device=self.device, dtype=torch.float32)

        if isinstance(state, int) or len(state) < self.state_dim:
            state_torch = to_one_hot_encoding(state, self.state_dim)
            next_state_torch = to_one_hot_encoding(next_state, self.state_dim)
        else:
            state_torch = torch.tensor(state, device=self.device, dtype=torch.float32)
            next_state_torch = torch.tensor(next_state, device=self.device, dtype=torch.float32)

        if self.reward_env_type == 0:
            reward_res = reward_torch

        elif self.reward_env_type == 1:
            reward_res = self.gamma*self.reward_net(next_state_torch) - self.reward_net(state_torch)

        elif self.reward_env_type == 2:
            reward_res = reward_torch + self.gamma*self.reward_net(next_state_torch) - self.reward_net(state_torch)

        elif self.reward_env_type == 3 or self.reward_env_type == 4:
            if not info:
                raise ValueError('No info dict provided by environment')

            info_torch = torch.tensor(list(info.values()), device=self.device, dtype=torch.float32)
            input_state = torch.cat((state_torch.to(self.device), info_torch.to(self.device)), dim=state_torch.dim() - 1)
            input_state_next = torch.cat((next_state_torch.to(self.device), info_torch.to(self.device)), dim=state_torch.dim() - 1)

            if self.reward_env_type == 3:
                reward_res = self.gamma * self.reward_net(input_state_next) - self.reward_net(input_state)
            elif self.reward_env_type == 4:
                reward_res = reward_torch + self.gamma * self.reward_net(input_state_next) - self.reward_net(input_state)

        elif self.reward_env_type == 5:
            reward_res = reward_torch + self.reward_net(next_state_torch)

        elif self.reward_env_type == 101 or self.reward_env_type == 102:
            if not info:
                raise ValueError('No info dict provided by environment')

            info_torch = torch.tensor(list(info.values()), device=self.device, dtype=torch.float32)

            if self.reward_env_type == 101:
                reward_res = torch.sum(info_torch*self.reward_net)
            elif self.reward_env_type == 102:
                reward_res = reward_torch + torch.sum(info_torch*self.reward_net)

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


    def set_agent_params(self, gamma):
        self.gamma = gamma

