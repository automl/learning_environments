import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import torch
from os import path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, **kwargs):
        self.max_speed = kwargs["max_speed"]
        self.max_torque = kwargs["max_torque"]
        self.g = kwargs["g"]
        self.m = kwargs["m"]
        self.l = kwargs["l"]
        self.dt = .05
        self._max_episode_steps=500
        self.viewer = None

        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = torch.clamp(u, -self.max_torque, self.max_torque)
        self.last_u = u.cpu().data.numpy()  # for rendering
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        newthdot = thdot + (-3 * g / (2 * l) * torch.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = torch.clamp(newthdot, -self.max_speed, self.max_speed)

        self.state = torch.stack((newth, newthdot), dim=0)
        done = torch.tensor([0], device=device, dtype=torch.float32)
        reward = torch.tensor([-costs], device=device, dtype=torch.float32)
        return self._get_obs(), reward, done, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = torch.tensor(self.np_random.uniform(low=-high, high=high), device=device, dtype=torch.float32)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        return torch.stack((torch.cos(self.state[0]),
                            torch.sin(self.state[0]),
                            self.state[1]), dim=0).squeeze()

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            #fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            #self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            #self.img.add_attr(self.imgtrans)

        #self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)