import copy
import numpy as np
from agents.TD3 import TD3
from agents.TD3_mod import TD3_Mod
from agents.PPO import PPO
from agents.DDQN import DDQN
from envs.env_factory import EnvFactory
from utils import print_abs_param_sum


def select_agent(config, agent_name):
    env_factory = EnvFactory(config)
    dummy_env = env_factory.generate_default_real_env()
    state_dim = dummy_env.get_state_dim()
    action_dim = dummy_env.get_action_dim()

    if agent_name == "TD3":
        max_action = dummy_env.get_max_action()
        return TD3(state_dim, action_dim, max_action, config)
    elif agent_name == "PPO":
        return PPO(state_dim, action_dim, config)
    elif agent_name == "DDQN":
        return DDQN(state_dim, action_dim, config)
    else:
        raise NotImplementedError("Unknownn RL agent")

def select_mod(config):
    env_factory = EnvFactory(config)
    dummy_env = env_factory.generate_default_real_env()
    state_dim = dummy_env.get_state_dim()
    action_dim = dummy_env.get_action_dim()
    max_action = dummy_env.get_max_action()

    return TD3_Mod(state_dim, action_dim, max_action, config)


def print_stats(agent):
    if isinstance(agent, TD3):
        print_abs_param_sum(agent.actor, "Actor")
        print_abs_param_sum(agent.critic_1, "Critic1")
        print_abs_param_sum(agent.critic_2, "Critic2")
    elif isinstance(agent, DDQN):
        print_abs_param_sum(agent.model, "Model")


def test(agent, env_factory, config, mod=None):
    # generate 6 different deterministic environments with increasing difficulty
    # and check for every environment how many episodes it takes the agent to solve it
    # N.B. we have to reset the state of the agent before every iteration

    # to avoid problems with wrongly initialized optimizers
    agent.reset_optimizer()

    mean_episodes_till_solved = 0
    episodes_till_solved = []
    agent_state = copy.deepcopy(agent.get_state_dict())

    if config['env_name'] == 'HalfCheetah-v2':
        interpolate_vals = [0, 0.03, 0.1, 0.4, 1]
    else:
        interpolate_vals = np.arange(0, 1.01, 0.2)

    for interpolate in interpolate_vals:
        agent.reset_optimizer()
        agent.set_state_dict(agent_state)
        if mod is not None:
            mod.set_mod_type(0)    # deactivate mod
        print_stats(agent=agent)
        env = env_factory.generate_interpolated_real_env(interpolate)
        reward_list = agent.train(env=env, mod=mod)
        mean_episodes_till_solved += len(reward_list)
        episodes_till_solved.append(len(reward_list))
        print("episodes till solved: " + str(len(reward_list)))

    agent.set_state_dict(agent_state)
    mean_episodes_till_solved /= len(interpolate_vals)

    return mean_episodes_till_solved, episodes_till_solved