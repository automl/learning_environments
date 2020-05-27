import numpy as np
import torch
import gym
import yaml
import time
from utils import ReplayBuffer, AverageMeter
from agents.PPO import PPO

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_PPO(config):
    ppo_config = config['ppo']
    env_name = config['env_name']
    seed = config['seed']
    max_episodes = ppo_config['max_episodes']
    update_episodes = ppo_config['update_episodes']

    # generate environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # set seeds
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # intialize RL learner
    ppo = PPO(state_dim = state_dim,
              action_dim = action_dim,
              config = config)

    replay_buffer = ReplayBuffer(state_dim, action_dim)
    avg_meter = AverageMeter(buffer_size = 100,
                             update_rate = 100,
                             print_str = 'Average reward: ')

    time_step = 0

    # training loop
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0

        for t in range(env._max_episode_steps):
            time_step += 1

            # run old policy
            action = ppo.select_action(state)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, next_state, reward, done)
            state = next_state

            episode_reward += reward

            # train after certain amount of timesteps
            if time_step/env._max_episode_steps > update_episodes:
                ppo.train(replay_buffer)
                replay_buffer.clear()
                time_step = 0
            if done:
                break

        # logging
        avg_meter.update(episode_reward)
        #print('Episode {} \t Reward: {}'.format(episode, episode_reward))


if __name__ == "__main__":
    with open("default_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    run_PPO(config)