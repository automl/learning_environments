from agents.TD3 import TD3
from agents.TD3_mod import TD3_Mod
from agents.PPO import PPO
from envs.env_factory import EnvFactory

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
