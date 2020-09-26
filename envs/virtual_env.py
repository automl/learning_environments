import torch
import torch.nn as nn
from models.model_utils import build_nn_from_config
from utils import from_one_hot_encoding, to_one_hot_encoding

class VirtualEnv(nn.Module):
    def __init__(self, kwargs):
        super().__init__()

        self.input_seed = None
        self.env_name = str(kwargs["env_name"])
        self.device = str(kwargs["device"])
        self.state_dim = int(kwargs["state_dim"])
        self.action_dim = int(kwargs["action_dim"])
        self.solved_reward = int(kwargs["solved_reward"])

        # for gym compatibility
        self._max_episode_steps = int(kwargs["max_steps"])
        self.action_space = kwargs["action_space"]
        self.observation_space = kwargs["observation_space"]
        self.reset_env = kwargs["reset_env"]

        self.state_net = build_nn_from_config(input_dim=self.state_dim + self.action_dim,
                                              output_dim=self.state_dim,
                                              nn_config=kwargs).to(self.device)
        # self.reward_net = build_nn_from_config(input_dim=self.state_dim + self.action_dim,
        #                                        output_dim=1,
        #                                        nn_config=kwargs,
        #                                        rbf_net=True,
        #                                        final_bias=-1).to(self.device)
        self.reward_net = build_nn_from_config(input_dim=self.state_dim + self.action_dim,
                                               output_dim=1,
                                               nn_config=kwargs).to(self.device)
        self.done_net = build_nn_from_config(input_dim=self.state_dim + self.action_dim,
                                             output_dim=1,
                                             nn_config=kwargs).to(self.device)

        self.state = self.reset()

    def reset(self):
        # dist = torch.distributions.Normal(torch.zeros(self.state_dim), torch.ones(self.state_dim))
        # self.state = dist.sample()
        # FIXME
        # self.state = torch.zeros(self.state_dim)
        # self.state[0] = -0.6 + torch.rand(1)*0.2
        self.state = torch.tensor(self.reset_env.reset())
        if len(self.state) > self.state_dim:
            self.state = from_one_hot_encoding(self.state)
        elif len(self.state) < self.state_dim:
            self.state = to_one_hot_encoding(self.state, self.state_dim)
        return self.state

    def step(self, action, state=None):
        if state is None:
            input = torch.cat((action.to(self.device), self.state.to(self.device)), dim=action.dim() - 1)
        else:
            input = torch.cat((action.to(self.device), state.to(self.device)), dim=action.dim() - 1)
        next_state = self.state_net(input)
        reward = self.reward_net(input)
        done = self.done_net(input)
        self.state = next_state
        return next_state, reward, done

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
