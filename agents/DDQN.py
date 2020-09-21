import yaml
import torch
import torch.nn as nn
import time
import random
import torch.nn.functional as F
from models.actor_critic import Critic_DQN
from utils import ReplayBuffer, AverageMeter, time_is_up, env_solved, print_abs_param_sum
from envs.env_factory import EnvFactory

class DDQN(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super().__init__()

        agent_name = "ddqn"
        ddqn_config = config["agents"][agent_name]

        self.train_episodes = ddqn_config["train_episodes"]
        self.test_episodes = ddqn_config["test_episodes"]
        self.init_episodes = ddqn_config["init_episodes"]
        self.batch_size = ddqn_config["batch_size"]
        self.same_action_num = ddqn_config["same_action_num"]
        self.gamma = ddqn_config["gamma"]
        self.lr = ddqn_config["lr"]
        self.tau = ddqn_config["tau"]
        self.eps = ddqn_config["eps_init"]
        self.eps_init = ddqn_config["eps_init"]
        self.eps_min = ddqn_config["eps_min"]
        self.eps_decay = ddqn_config["eps_decay"]
        self.rb_size = ddqn_config["rb_size"]
        self.print_rate = ddqn_config["print_rate"]
        self.early_out_num = ddqn_config["early_out_num"]
        self.early_out_virtual_diff = ddqn_config["early_out_virtual_diff"]
        self.render_env = config["render_env"]
        self.device = config["device"]

        self.model = Critic_DQN(state_dim, action_dim, agent_name, config).to(self.device)
        self.model_target = Critic_DQN(state_dim, action_dim, agent_name, config).to(self.device)
        self.model_target.load_state_dict(self.model.state_dict())

        self.reset_optimizer()

        self.it = 0


    def train(self, env, time_remaining=1e9):
        #print_abs_param_sum(self.model, name='DDQN model weight before: ')

        time_start = time.time()

        replay_buffer = ReplayBuffer(state_dim=env.get_state_dim(), action_dim=1, device=self.device, max_size=self.rb_size)
        avg_meter_reward = AverageMeter(print_str="Average reward: ")
        avg_meter_eps = AverageMeter(print_str="Average eps: ")

        self.eps = self.eps_init

        # training loop
        for episode in range(self.train_episodes):
            # early out if timeout
            if time_is_up(avg_meter_reward=avg_meter_reward,
                          max_episodes=self.train_episodes,
                          time_elapsed=time.time()-time_start,
                          time_remaining=time_remaining):
                break

            state = env.reset()
            episode_reward = 0

            for t in range(0, env.max_episode_steps(), self.same_action_num):
                if random.random() < self.eps:
                    action = env.get_random_action()
                else:
                    qvals = self.model(state.to(self.device))
                    action = torch.argmax(qvals).unsqueeze(0).detach()

                # live view
                if self.render_env and episode % 10 == 0:
                    env.render()

                # state-action transition
                next_state, reward, done = env.step(action=action, same_action_num=self.same_action_num)
                replay_buffer.add(state=state, action=action, next_state=next_state, reward=reward, done=done)
                state = next_state
                episode_reward += reward

                # train
                if episode > self.init_episodes:
                    self.learn(replay_buffer)

                if done > 0.5:
                    break

            # logging
            avg_meter_reward.update(episode_reward, print_rate=self.print_rate)
            avg_meter_eps.update(self.eps, print_rate=self.print_rate)

            # update eps
            self.eps *= self.eps_decay
            self.eps = max(self.eps, self.eps_min)

            # quit training if environment is solved
            if env_solved(agent=self, env=env, avg_meter_reward=avg_meter_reward, episode=episode):
                break

        env.close()

        #print_abs_param_sum(self.model, name='DDQN model weight after:  ')

        return avg_meter_reward.get_raw_data()


    def learn(self, replay_buffer):
        self.it += 1

        states, actions, next_states, rewards, dones = replay_buffer.sample(self.batch_size)

        states = states.squeeze()
        actions = actions.squeeze()
        next_states = next_states.squeeze()
        rewards = rewards.squeeze()
        dones = dones.squeeze()

        if len(states.shape) == 1:
            states = states.unsqueeze(1)
            next_states = next_states.unsqueeze(1)

        q_values = self.model(states)
        next_q_values = self.model(next_states)
        next_q_state_values = self.model_target(next_states)

        q_value = q_values.gather(1, actions.long().unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, next_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = rewards + self.gamma * next_q_value * (1 - dones)
        loss = F.mse_loss(q_value, expected_q_value.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # target network update
        for target_param, param in zip(self.model_target.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
        return loss


    def test(self, env, time_remaining=1e9):
        with torch.no_grad():
            time_start = time.time()

            avg_meter_reward = AverageMeter(print_str="Average reward: ")

            # training loop
            for episode in range(self.test_episodes):
                # early out if timeout
                if time_is_up(avg_meter_reward=avg_meter_reward,
                              max_episodes=self.test_episodes,
                              time_elapsed=time.time() - time_start,
                              time_remaining=time_remaining):
                    break

                state = env.reset()
                episode_reward = 0

                for t in range(0, env.max_episode_steps(), self.same_action_num):
                    qvals = self.model(state.to(self.device))
                    action = torch.argmax(qvals).unsqueeze(0).detach()

                    # if t == 0:
                    #     print(action)

                    # live view
                    if self.render_env and episode % 10 == 0:
                        env.render()

                    # state-action transition
                    next_state, reward, done = env.step(action=action, same_action_num=self.same_action_num)
                    state = next_state
                    episode_reward += reward

                    if done > 0.5:
                        break

                # logging
                avg_meter_reward.update(episode_reward, print_rate=self.print_rate)

                # quit training if environment is solved
                if env_solved(agent=self, env=env, avg_meter_reward=avg_meter_reward, episode=episode):
                    break

            env.close()

        return avg_meter_reward.get_raw_data()


    def reset_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)


    def get_state_dict(self):
        agent_state = {}
        agent_state["ddqn_model"] = self.model.state_dict()
        agent_state["ddqn_model_target"] = self.model_target.state_dict()
        if self.optimizer:
            agent_state["ddqn_optimizer"] = self.optimizer.state_dict()

        return agent_state

    def set_state_dict(self, agent_state):
        self.model.load_state_dict(agent_state["ddqn_model"])
        self.model_target.load_state_dict(agent_state["ddqn_model_target"])
        if "ddqn_optimizer" in agent_state.keys():
            self.optimizer.load_state_dict(agent_state["ddqn_optimizer"])


if __name__ == "__main__":
    with open("../default_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    # seed = config["seed"]
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    # generate environment
    env_fac = EnvFactory(config)
    virt_env = env_fac.generate_virtual_env()
    real_env = env_fac.generate_default_real_env()

    # ddqn = DDQN(state_dim=virt_env.get_state_dim(),
    #             action_dim=virt_env.get_action_dim(),
    #             config=config)
    mean_reward_list = []

    ddqn = DDQN(state_dim=virt_env.get_state_dim(),
                action_dim=virt_env.get_action_dim(),
                config=config)

    #ddqn.train(env=virt_env, time_remaining=50)
    for i in range(1000):
        print(i)

        ddqn = DDQN(state_dim=virt_env.get_state_dim(),
                    action_dim=virt_env.get_action_dim(),
                    config=config)

        ddqn.train(env=real_env, time_remaining=50)
        reward_list = ddqn.test(env=real_env, time_remaining=50)
        print(sum(reward_list)/len(reward_list))
    #     mean_reward_list.append(sum(reward_list)/len(reward_list))
    #     print(sum(mean_reward_list) / len(mean_reward_list))
    #
    # print(sum(mean_reward_list)/len(mean_reward_list))


