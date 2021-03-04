import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import build_nn_from_config


class ICMModel(nn.Module):
    def __init__(self, state_dim, action_dim, feature_dim=64, hidden_size=128, act_1="leakyrelu", act_2="relu", act_3="identity"):
        super(ICMModel, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim

        nn_features_config = {
                'hidden_size': hidden_size,
                'hidden_layer': 2,
                'activation_fn': act_1
                }
        nn_inverse_config = {
                'hidden_size': hidden_size,
                'hidden_layer': 2,
                'activation_fn': act_2
                }
        nn_forward_pre_config = {
                'hidden_size': hidden_size,
                'hidden_layer': 2,
                'activation_fn': act_1
                }
        nn_forward_post_config = {
                'hidden_size': hidden_size,
                'hidden_layer': 1,
                'activation_fn': act_3
                }

        self.features_model = build_nn_from_config(input_dim=state_dim, output_dim=feature_dim, nn_config=nn_features_config)
        self.inverse_model = build_nn_from_config(input_dim=feature_dim * 2, output_dim=action_dim, nn_config=nn_inverse_config)

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


class ICM:
    """ Intrinsic Curiosity Module """

    def __init__(self, state_dim, action_dim, feature_dim=64, hidden_size=128, learning_rate=1e-4, beta=.2, eta=.5, act_1="leakyrelu",
                 act_2="relu", act_3="identity", device="cpu"):
        self.device = device
        self.model = ICMModel(state_dim=state_dim, action_dim=action_dim, feature_dim=feature_dim, hidden_size=hidden_size, act_1=act_1,
                              act_2=act_2, act_3=act_3).to(self.device)
        self.beta = beta
        self.lr = learning_rate
        self.eta = eta

        icm_params = list(self.model.parameters())
        self.icm_optimizer = torch.optim.Adam(icm_params, lr=self.lr)

    def train(self, states, next_states, actions):
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(dim=1)
        next_states_encoded, next_states_pred_encoded, actions_pred = self.model(input=(states, next_states, actions))

        # compute ICM loss
        icm_loss = (1 - self.beta) * F.mse_loss(actions_pred, actions) + self.beta * F.mse_loss(next_states_encoded,
                                                                                                next_states_pred_encoded)

        # Optimize ICM
        self.icm_optimizer.zero_grad()
        icm_loss.backward()
        self.icm_optimizer.step()

    def compute_intrinsic_rewards(self, state, next_state, action):
        if len(action.shape) == 1:
            action = action.unsqueeze(dim=1)
        next_states_encoded, next_states_pred_encoded, actions_pred = self.model(input=(state, next_state, action))
        intrinsic_reward = self.eta * F.mse_loss(next_states_encoded, next_states_pred_encoded, reduction="none").mean(-1)
        return intrinsic_reward.detach().unsqueeze(dim=1)
