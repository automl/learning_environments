import pytest
import numpy as np
import torch
from agents.TD3 import TD3
from envs.env_factory import EnvFactory

def create_td3_config_base(init_episodes, max_episodes):
    cfg = {}
    cfg['env_name'] = 'Pendulum-v0'
    cfg['seed'] = 0
    cfg['render_env'] = False

    cfg_td3 = {}
    cfg_td3['init_episodes'] = init_episodes
    cfg_td3['max_episodes'] = max_episodes
    cfg_td3['batch_size'] = 256
    cfg_td3['gamma'] = 0.99
    cfg_td3['lr'] = 3e-4
    cfg_td3['weight_decay'] = 0
    cfg_td3['tau'] = 0.005
    cfg_td3['policy_delay'] = 2
    cfg_td3['rb_size'] = 100000
    cfg_td3['hidden_size'] = 256
    cfg_td3['activation_fn'] = 'relu'
    cfg_td3['action_std'] = 0.1
    cfg_td3['optim_env_with_ac'] = 0
    cfg_td3['early_out_num'] = 1000
    cfg_td3['weight_norm'] = True
    cfg_td3['optim_env_with_actor'] = False
    cfg_td3['optim_env_with_critic'] = False
    cfg_td3['match_weight_actor'] = 0
    cfg_td3['match_weight_critic'] = 0
    cfg_td3['match_batch_size'] = 0
    cfg['agents'] = {}
    cfg['agents']['td3'] = cfg_td3

    cfg_pen = {}
    cfg_pen['max_steps'] = 200
    cfg_pen['solved_reward'] = 0
    cfg['envs'] = {}
    cfg['envs']['Pendulum-v0'] = cfg_pen

    return cfg

def run_td3(init_episodes, max_episodes):
    config = create_td3_config_base(init_episodes=init_episodes,
                                    max_episodes=max_episodes)
    seed = config['seed']

    # generate environment
    env_fac = EnvFactory(config)
    env = env_fac.generate_default_real_env()

    # set seeds
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    td3 = TD3(state_dim=env.get_state_dim(),
              action_dim=env.get_action_dim(),
              config=config)

    return td3.run(env)


def test_td3_quick():
    # quick test to check if something crashes
    episode_rewards = run_td3(init_episodes=3,
                              max_episodes=6)

    for reward in episode_rewards:
        assert reward > -2000
        assert reward < -500


@pytest.mark.slow
def test_td3_complete():
    # more mature test to check if TD3 still works properly
    episode_rewards = run_td3(init_episodes=100,
                              max_episodes=200)

    for i in range(len(episode_rewards)):
        reward = episode_rewards[i]

        if i < 100:
            assert reward > -2000
            assert reward < -500
        elif i > 190:
            assert reward > 500
            assert reward < 0
