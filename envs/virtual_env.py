import gym
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.weight_norm import weight_norm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VirtualEnv(nn.Module, gym.Env):
    def __init__(self, state_dim, action_dim, **kwargs):
        super(VirtualEnv, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_seed = None
        self.state = torch.zeros(state_dim, device="cpu")  # not sure what the default state should be
        self._max_episode_steps = int(kwargs["max_steps"])

        self.viewer = None

        hidden_size = kwargs["hidden_size"]
        apply_weight_norm = kwargs["weight_norm"]

        self.virtual_env_zero_init = kwargs["virtual_env_zero_init"]
        self.env_name = kwargs["env_name"]

        if apply_weight_norm:
            self.base = nn.Sequential(
                weight_norm(nn.Linear(state_dim + action_dim + 1, hidden_size)),  # +1 because of seed
                nn.Tanh(),
                weight_norm(nn.Linear(hidden_size, hidden_size)),
                nn.Tanh(),
            )
            self.state_head = weight_norm(nn.Linear(hidden_size, state_dim))
            self.reward_head = weight_norm(nn.Linear(hidden_size, 1))
            self.done_head = weight_norm(nn.Linear(hidden_size, 1))
        else:
            self.base = nn.Sequential(
                nn.Linear(state_dim + action_dim + 1, hidden_size),  # +1 because of seed
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
            )
            self.state_head = nn.Linear(hidden_size, state_dim)
            self.reward_head = nn.Linear(hidden_size, 1)
            self.done_head = nn.Linear(hidden_size, 1)

    def reset(self):
        if self.virtual_env_zero_init:
            self.state = torch.zeros(self.state_dim, device="cpu")
        elif self.env_name == "PendulumEnv":
            self.state = torch.tensor(
                [
                    np.random.uniform(low=-np.pi, high=np.pi),
                    np.random.uniform(low=-1, high=1),
                    np.random.uniform(low=-8, high=8),
                ],
                device="cpu",
                dtype=torch.float32,
            )
        else:
            raise NotImplementedError("Unknown environment: non-zero state reset only for supported environments.")
        self.last_action = None

        return self.state

    def set_seed(self, seed):
        self.input_seed = seed

    def set_state(self, state):
        self.state = state

    def step(self, action):
        if not self.input_seed:
            raise ValueError("no input seed set")

        # used only for rendering
        self.last_action = action.cpu().data.numpy()

        input = torch.cat((action, self.state, self.input_seed))
        x = self.base(input)
        next_state = self.state_head(x)
        reward = self.reward_head(x)
        done = self.done_head(x) > 0.5
        self.state = next_state
        return next_state, reward, done, {}

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, 0.2)
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            # fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            # self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()  # self.img.add_attr(self.imgtrans)

        # self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_action:
            self.imgtrans.scale = (-self.last_action / 2, np.abs(self.last_action) / 2)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
