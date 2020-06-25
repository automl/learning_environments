from agents.TD3 import TD3
from agents.PPO import PPO
from envs.env_factory import EnvFactory

def select_agent(config, agent_name):
    env_factory = EnvFactory(config)
    dummy_env = env_factory.generate_default_real_env()
    state_dim = dummy_env.get_state_dim()
    action_dim = dummy_env.get_action_dim()

    if agent_name == "TD3":
        return TD3(state_dim, action_dim, config)
    elif agent_name == "PPO":
        return PPO(state_dim, action_dim, config)
    else:
        raise NotImplementedError("Unknownn RL agent")


