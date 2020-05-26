import numpy as np
import torch
import gym
import yaml
from utils import ReplayBuffer
from agents.TD3 import TD3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_TD3(config):
    env_name = config['env_name']
    seed = config['seed']
    td3_config = config['td3']
    init_episodes = td3_config['init_episodes']
    max_episodes = td3_config['max_episodes']
    rb_size = td3_config['rb_size']

    # generate environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # set seeds
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    td3 = TD3(state_dim = state_dim,
              action_dim = action_dim,
              config = config)

    replay_buffer = ReplayBuffer(state_dim, action_dim, rb_size)

    time_step = 0

    # training loop
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0

        for t in range(env._max_episode_steps):
            time_step += 1

            # fill replay buffer at beginning
            if episode < init_episodes:
                action = env.action_space.sample()
            else:
                action = td3.select_action(state)

            # state-action transition
            next_state, reward, done, _ = env.step(action)
            done_bool = float(done) if t < env._max_episode_steps-1 else 0
            replay_buffer.add(state, action, next_state, reward, done_bool)
            state = next_state

            episode_reward += reward

            # train
            if episode > init_episodes:
                td3.train(replay_buffer)
            if done:
                break

        # logging
        print('Episode {} \t Reward: {}'.format(episode, episode_reward))



if __name__ == "__main__":
    with open("default_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    run_TD3(config)