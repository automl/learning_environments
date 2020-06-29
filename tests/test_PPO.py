import pytest
import numpy as np
import torch
from agents.PPO import PPO
from envs.env_factory import EnvFactory

def create_ppo_config_base(max_episodes):
    cfg = {}
    cfg['env_name'] = 'Pendulum-v0'
    cfg['seed'] = 0
    cfg['render_env'] = False

    cfg_ppo = {}
    cfg_ppo['max_episodes'] = max_episodes
    cfg_ppo['update_episodes'] = 20
    cfg_ppo['ppo_epochs'] = 100
    cfg_ppo['gamma'] = 0.99
    cfg_ppo['lr'] = 5e-4
    cfg_ppo['weight_decay'] = 0
    cfg_ppo['vf_coef'] = 0.5
    cfg_ppo['ent_coef'] = 0.01
    cfg_ppo['eps_clip'] = 0.2
    cfg_ppo['hidden_size'] = 64
    cfg_ppo['activation_fn'] = 'relu'
    cfg_ppo['action_std'] = 0.5
    cfg_ppo['early_out_num'] = 10000
    cfg['agents'] = {}
    cfg['agents']['ppo'] = cfg_ppo

    cfg_pen = {}
    cfg_pen['max_steps'] = 200
    cfg_pen['solved_reward'] = 0
    cfg['envs'] = {}
    cfg['envs']['Pendulum-v0'] = cfg_pen

    return cfg

def run_ppo(max_episodes):
    config = create_ppo_config_base(max_episodes=max_episodes)
    seed = config['seed']

    # generate environment
    env_fac = EnvFactory(config)
    env = env_fac.generate_default_real_env()

    # set seeds
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    ppo = PPO(state_dim=env.get_state_dim(),
              action_dim=env.get_action_dim(),
              config=config)

    return ppo.run(env)


def test_ppo_quick():
    # quick test to check if something crashes
    episode_rewards = run_ppo(max_episodes=50)

    for reward in episode_rewards:
        assert reward > -2000
        assert reward < -500


@pytest.mark.slow
def test_ppo_complete():
    # more mature test to check if TD3 still works properly
    episode_rewards = run_ppo(max_episodes=6000)

    for i in range(len(episode_rewards)):
        reward = episode_rewards[i]

        if i < 100:
            assert reward > -2000
            assert reward < -500
        elif i > 5900:
            assert reward > 500
            assert reward < 0
