import numpy as np
from gym.utils import seeding
from gym import utils
from gym.envs.mujoco import mujoco_env


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.cripple_mask = np.ones(6)  # required before mujoco env init by step()

        mujoco_env.MujocoEnv.__init__(self, "half_cheetah.xml", 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        action *= self.cripple_mask

        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos.flat[1:], self.sim.data.qvel.flat])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_kwargs_with_mujoco_support(self, kwargs):
        for key, value in kwargs.items():
            if "g" == key:  # gravity along negative z-axis
                self.model.opt.gravity[2] = value
            elif "cripple_joint" == key:
                if value:  # cripple_joint True
                    self.cripple_mask = np.ones(self.action_space.shape)
                    idx = np.random.choice(self.action_space.shape[0])
                    self.cripple_mask[idx] = 0
            else:
                setattr(self, key, value)

        # additional params to play with
        # self.sim.model.geom_friction
        # self.model.body_mass
        # self.sim.data.actuator_length