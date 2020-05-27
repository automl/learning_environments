from utils import Actor, Critic_Q
import copy
import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TD3(object):
    def __init__(self, state_dim, action_dim, config):

        td3_config = config['agents']['td3']

        self.gamma = td3_config['gamma']
        self.tau = td3_config['tau']
        self.policy_delay = td3_config['policy_delay']
        self.batch_size = td3_config['batch_size']

        self.device = device
        self.actor = Actor(state_dim, action_dim, config).to(device)
        self.actor_target = Actor(state_dim, action_dim, config).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=td3_config['lr'])

        self.critic_1 = Critic_Q(state_dim, action_dim, config).to(device)
        self.critic_2 = Critic_Q(state_dim, action_dim, config).to(device)
        self.critic_target_1 = Critic_Q(state_dim, action_dim, config).to(device)
        self.critic_target_2 = Critic_Q(state_dim, action_dim, config).to(device)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        self.critic_optimizer = torch.optim.Adam(list(self.critic_1.parameters()) +
                                                 list(self.critic_2.parameters()), lr=td3_config['lr'])

        self.total_it = 0


    def run(self, ):


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()


    def train(self, replay_buffer):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, done = replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            next_action = (self.actor_target(next_state))

            # Compute the target Q value
            target_Q1 = self.critic_target_1(next_state, next_action)
            target_Q2 = self.critic_target_2(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1-done) * self.gamma * target_Q

        # Get current Q estimates
        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_delay == 0:

            # Compute actor loss
            actor_loss = (-self.critic_1(state, self.actor(state))).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic_1.parameters(), self.critic_target_1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
