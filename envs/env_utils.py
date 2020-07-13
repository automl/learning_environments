from envs.pendulum import PendulumEnv
from envs.test_env import TestEnv
from envs.continuous_mountain_car import Continuous_MountainCarEnv
from envs.half_cheetah import HalfCheetahEnv


def generate_env_with_kwargs(kwargs, env_name):
    if env_name == "Pendulum-v0":
        # env = gym.make(self.env_name)
        env = PendulumEnv()
    elif env_name == "MountainCarContinuous-v0":
        env = Continuous_MountainCarEnv()
    elif env_name == "HalfCheetah-v2":
        env = HalfCheetahEnv()
    elif env_name == "Test":
        env = TestEnv()
    else:
        raise NotImplementedError("Environment not supported")

    if env_name == "HalfCheetah-v2":
        env.set_kwargs_with_mujoco_support(kwargs)
    else:
        for key, value in kwargs.items():
            setattr(env, key, value)

    # needed for stopping the episodes
    env._max_episode_steps = int(kwargs["max_steps"])
    # needed for model save/load
    env.kwargs = kwargs
    return env
