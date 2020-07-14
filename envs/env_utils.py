import numpy as np
from envs.pendulum import PendulumEnv
from envs.test_env import TestEnv
from envs.continuous_mountain_car import Continuous_MountainCarEnv
from envs.half_cheetah import HalfCheetahEnv


def generate_env_with_kwargs(kwargs, env_name):
    # generate environment class
    if env_name == "Pendulum-v0":
        env = PendulumEnv()
    elif env_name == "MountainCarContinuous-v0":
        env = Continuous_MountainCarEnv()
    elif env_name == "HalfCheetah-v2":
        env = HalfCheetahEnv()
    elif env_name == "Test":
        env = TestEnv()
    else:
        raise NotImplementedError("Environment not supported")

    # set environment parameters
    if env_name == "HalfCheetah-v2":
        for key, value in kwargs.items():
            if "g" == key:  # gravity along negative z-axis
                env.model.opt.gravity[2] = value
            elif "cripple_joint" == key:
                if value:  # cripple_joint True
                    env.cripple_mask = np.ones(env.action_space.shape)
                    idx = np.random.choice(env.action_space.shape[0])
                    env.cripple_mask[idx] = 0
            else:
                setattr(env, key, value)
    else:
        for key, value in kwargs.items():
            setattr(env, key, value)

    # for episode termination
    env._max_episode_steps = int(kwargs["max_steps"])
    # for model save/load
    env.kwargs = kwargs

    return env
