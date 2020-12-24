import torch.nn as nn
from models.model_utils import build_nn_from_config
from agents.TD3 import TD3
import yaml
import time
import torch
import torch.nn.functional as F
from envs.env_factory import EnvFactory


class ICMModel(nn.Module):
    def __init__(self, state_dim, action_dim, feature_dim=128):
        super(ICMModel, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim

        nn_features_config = {'hidden_size': 128, 'hidden_layer': 2, 'activation_fn': 'leakyrelu'}
        nn_inverse_config = {'hidden_size': 128, 'hidden_layer': 1, 'activation_fn': 'relu'}
        nn_forward_pre_config = {'hidden_size': 128, 'hidden_layer': 1, 'activation_fn': 'leakyrelu'}
        nn_forward_post_config = {'hidden_size': 128, 'hidden_layer': 1, 'activation_fn': 'identity'}

        self.features_model = build_nn_from_config(input_dim=state_dim, output_dim=feature_dim, nn_config=nn_features_config)
        self.inverse_model = build_nn_from_config(input_dim=feature_dim*2, output_dim=action_dim, nn_config=nn_inverse_config)

        self.forward_pre_model = build_nn_from_config(input_dim=action_dim + feature_dim,
                                                      output_dim=feature_dim,
                                                      nn_config=nn_forward_pre_config
                                                      )

        class ResidualBlock(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.fc1 = nn.Sequential(
                                nn.Linear(input_dim, output_dim),
                                nn.LeakyReLU(inplace=True),
                           )
                self.fc2 = nn.Sequential(
                                nn.Linear(input_dim, output_dim)
                           )

            def forward(self, feature, action):
                x = feature
                x = self.fc1(torch.cat([x, action], dim=1))
                x = self.fc2(torch.cat([x, action], dim=1))
                return feature + x

        # original implementation uses residual blocks:
        # https://github.com/openai/large-scale-curiosity/blob/master/dynamics.py#L55-L61
        self.residual_block1 = ResidualBlock(input_dim=action_dim + feature_dim, output_dim=feature_dim)
        self.residual_block2 = ResidualBlock(input_dim=action_dim + feature_dim, output_dim=feature_dim)
        self.residual_block3 = ResidualBlock(input_dim=action_dim + feature_dim, output_dim=feature_dim)
        self.residual_block4 = ResidualBlock(input_dim=action_dim + feature_dim, output_dim=feature_dim)

        self.forward_post_model = build_nn_from_config(input_dim=feature_dim, output_dim=feature_dim, nn_config=nn_forward_post_config)

    def forward(self, input):
        state, next_state, action = input

        state_encoded = self.features_model(state)
        next_state_encoded = self.features_model(next_state)

        # get predicted action from inverse model
        state_next_state_concat = torch.cat((state_encoded, next_state_encoded), 1)
        action_pred = self.inverse_model(state_next_state_concat)

        # get predicted next state encoding from forward model with residual connections
        forward_model_input = torch.cat([state_encoded, action], 1)
        x = self.forward_pre_model(forward_model_input)

        x = self.residual_block1(feature=x, action=action)
        x = self.residual_block2(feature=x, action=action)
        x = self.residual_block3(feature=x, action=action)
        x = self.residual_block4(feature=x, action=action)

        next_state_pred_encoded = self.forward_post_model(x)

        return next_state_encoded, next_state_pred_encoded, action_pred


class ICMTD3(TD3):
    """
    TD3 with an Intrinsic Curiosity Module ("Curiosity-driven Exploration by Self-supervised Prediction", Pathak et al.)
    """
    def __init__(self, env, max_action, config, agent_name=None):
        super(ICMTD3, self).__init__(env=env, max_action=max_action, config=config, agent_name=agent_name)

        # from the paper:
        self.beta = .2
        self.lambd = .1
        #self.icm_lr = 1e-3
        self.icm = ICMModel(state_dim=self.state_dim, action_dim=self.action_dim, feature_dim=128).to(self.device)

        # overwrite actor optimizer, could also add ICM loss to critic loss
        params = list(self.icm.parameters()) + list(self.actor.parameters())
        #self.icm_optimizer = torch.optim.Adam(icm_params, lr=self.lr)
        self.actor_optimizer = torch.optim.Adam(params, lr=self.lr)

    def learn(self, replay_buffer, env, episode):
        self.total_it += 1

        # Sample replay buffer
        states, actions, next_states, rewards, dones = replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise, no_grad since target will be copied
            noise = (torch.randn_like(actions) * self.policy_std
                     ).clamp(-self.policy_std_clip, self.policy_std_clip)
            next_actions = (self.actor_target(next_states) + noise
                            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1 = self.critic_target_1(next_states, next_actions)
            target_Q2 = self.critic_target_2(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * self.gamma * target_Q
            # target_Q = rewards + self.gamma * target_Q

        # Get current Q estimates
        current_Q1 = self.critic_1(states, actions)
        current_Q2 = self.critic_2(states, actions)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_delay == 0:
            # Compute actor loss
            # todo: check algorithm 1 in original paper; has additional multiplicative term here
            actor_loss = (-self.critic_1(states, self.actor(states))).mean()

            # ICM part
            next_states_encoded, next_states_pred_encoded, actions_pred = self.icm(input=(states, next_states, actions))

            # compute ICM loss
            icm_loss = (1 - self.beta) * F.mse_loss(actions_pred, actions) + \
                       self.beta * F.mse_loss(next_states_encoded, next_states_pred_encoded)

            actor_loss = self.lambd * actor_loss + icm_loss

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


if __name__ == "__main__":
    with open("../default_config_pendulum.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    # generate environment
    env_fac = EnvFactory(config)
    real_env= env_fac.generate_real_env()
    td3 = ICMTD3(env=real_env,
              max_action=real_env.get_max_action(),
              config=config)
    t1 = time.time()
    td3.train(env=real_env, time_remaining=1200)
    print(time.time()-t1)
    #td3.train(env=virt_env, time_remaining=5)