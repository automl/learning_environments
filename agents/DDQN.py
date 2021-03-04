import random

import torch
import torch.nn.functional as F
import yaml

from agents.base_agent import BaseAgent
from envs.env_factory import EnvFactory
from models.actor_critic import Critic_DQN
from models.icm_baseline import ICM
from utils import to_one_hot_encoding


class DDQN(BaseAgent):
    def __init__(self, env, config, icm=False):
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

        self.icm = None
        if icm:
            icm_config = config["agents"]["icm"]
            self.icm_lr = icm_config["learning_rate"]
            self.icm_beta = icm_config["beta"]
            self.icm_eta = icm_config["eta"]
            self.icm_feature_dim = icm_config["feature_dim"]
            self.icm_hidden_dim = icm_config["hidden_size"]
            actual_action_dim = len(env.get_random_action())
            self.icm = ICM(state_dim=self.state_dim,
                           action_dim=actual_action_dim,
                           learning_rate=self.icm_lr,
                           beta=self.icm_beta,
                           eta=self.icm_eta,
                           feature_dim=self.icm_feature_dim,
                           hidden_size=self.icm_hidden_dim,
                           device=self.device)

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

        if self.icm:
            self.icm.train(states, next_states, actions)
            rewards += self.icm.compute_intrinsic_rewards(states, next_states, actions).squeeze()

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
            return torch.argmax(qvals).unsqueeze(0).detach()

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
    with open("../default_config_cartpole.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    torch.set_num_threads(1)

    # generate environment
    env_fac = EnvFactory(config)
    virt_env = env_fac.generate_virtual_env()
    real_env = env_fac.generate_real_env()

    timing = []
    for i in range(10):
        ddqn = DDQN(env=real_env,
                    config=config,
                    icm=True)

        # ddqn.train(env=virt_env, time_remaining=50)

        print('TRAIN')
        ddqn.train(env=real_env, time_remaining=500)
        print('TEST')
        ddqn.test(env=real_env, time_remaining=500)
