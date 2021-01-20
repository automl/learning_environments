import yaml
import torch
import time
import random
import torch.nn.functional as F
from agents.base_agent import BaseAgent
from models.actor_critic import Critic_DQN
from envs.env_factory import EnvFactory
from utils import ReplayBuffer, AverageMeter, to_one_hot_encoding


class DDQN(BaseAgent):
    def __init__(self, env, config):
        self.agent_name = "ddqn"

        super().__init__(agent_name=self.agent_name, env=env, config=config)

        ddqn_config = config["agents"][self.agent_name]

        self.batch_size = ddqn_config["batch_size"]
        self.rb_size = ddqn_config["rb_size"]
        self.gamma = ddqn_config["gamma"]
        self.lr = ddqn_config["lr"]
        self.tau = ddqn_config["tau"]
        self.eps = ddqn_config["eps_init"]
        self.eps_init = ddqn_config["eps_init"]
        self.eps_min = ddqn_config["eps_min"]
        self.eps_decay = ddqn_config["eps_decay"]

        self.model = Critic_DQN(self.state_dim, self.action_dim, self.agent_name, config).to(self.device)
        self.model_target = Critic_DQN(self.state_dim, self.action_dim, self.agent_name, config).to(self.device)
        self.model_target.load_state_dict(self.model.state_dict())

        self.reset_optimizer()

        self.it = 0


    def learn(self, replay_buffer, env, episode):
        self.it += 1

        states, actions, next_states, rewards, dones = replay_buffer.sample(self.batch_size)

        states = states.squeeze()
        actions = actions.squeeze()
        next_states = next_states.squeeze()
        rewards = rewards.squeeze()
        dones = dones.squeeze()

        if env.has_discrete_state_space():
            states = to_one_hot_encoding(states, self.state_dim)
            next_states = to_one_hot_encoding(next_states, self.state_dim)

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


    def select_train_action(self, state, env, episode):
        if random.random() < self.eps:
            return env.get_random_action()
        else:
            if env.has_discrete_state_space():
                state = to_one_hot_encoding(state, self.state_dim)
            qvals = self.model(state.to(self.device))
            return  torch.argmax(qvals).unsqueeze(0).detach()


    def select_test_action(self, state, env):
        if env.has_discrete_state_space():
            state = to_one_hot_encoding(state, self.state_dim)
        qvals = self.model(state.to(self.device))
        return torch.argmax(qvals).unsqueeze(0).detach()


    def update_parameters_per_episode(self, episode):
        if episode == 0:
            self.eps = self.eps_init
        else:
            self.eps *= self.eps_decay
            self.eps = max(self.eps, self.eps_min)


    def reset_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)


if __name__ == "__main__":
    with open("../default_config_cartpole_opt.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    torch.set_num_threads(1)

    # generate environment
    env_fac = EnvFactory(config)
    virt_env = env_fac.generate_virtual_env()
    real_env = env_fac.generate_real_env()

    timing = []
    for i in range(10):
        ddqn = DDQN(env=real_env,
                    config=config)

        #ddqn.train(env=virt_env, time_remaining=50)

        print('TRAIN')
        ddqn.train(env=real_env, time_remaining=500)
        print('TEST')
        reward_list = ddqn.test(env=real_env, time_remaining=500)

