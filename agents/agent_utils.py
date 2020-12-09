from agents.TD3 import TD3
from agents.PPO import PPO
from agents.DDQN import DDQN
from agents.DDQN_vary import DDQN_vary
from agents.DuelingDDQN import DuelingDDQN
from agents.DuelingDDQN_vary import DuelingDDQN_vary
from agents.TD3_discrete_vary import TD3_discrete_vary
from agents.QL import QL
from agents.SARSA import SARSA
from envs.env_factory import EnvFactory
from utils import print_abs_param_sum

def select_agent(config, agent_name):
    env_factory = EnvFactory(config)
    dummy_env = env_factory.generate_real_env(print_str='Select Agent: ')

    agent_name = agent_name.lower()

    if agent_name == "td3":
        max_action = dummy_env.get_max_action()
        return TD3(env=dummy_env, max_action=max_action, config=config)
    elif agent_name == "ppo":
        return PPO(env=dummy_env, config=config)
    elif agent_name == "ddqn":
        return DDQN(env=dummy_env, config=config)
    elif agent_name == "ddqn_vary":
        return DDQN_vary(env=dummy_env, config=config)
    elif agent_name == "duelingddqn":
        return DuelingDDQN(env=dummy_env, config=config)
    elif agent_name == "duelingddqn_vary":
        return DuelingDDQN_vary(env=dummy_env, config=config)
    elif agent_name == "td3_discrete_vary":
        max_action = dummy_env.get_max_action()
        min_action = dummy_env.get_min_action()
        return TD3_discrete_vary(env=dummy_env, config=config, min_action=min_action, max_action=max_action)
    elif agent_name == "ql":
        return QL(env=dummy_env, config=config)
    elif agent_name == "sarsa":
        return SARSA(env=dummy_env, config=config)
    else:
        raise NotImplementedError("Unknownn RL agent")


def print_stats(agent):
    if isinstance(agent, TD3):
        print_abs_param_sum(agent.actor, "Actor")
        print_abs_param_sum(agent.critic_1, "Critic1")
        print_abs_param_sum(agent.critic_2, "Critic2")
    elif isinstance(agent, DDQN):
        print_abs_param_sum(agent.model, "Model")