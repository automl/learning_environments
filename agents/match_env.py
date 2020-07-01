import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.REPTILE import reptile_update
from utils import ReplayBuffer, AverageMeter, print_abs_param_sum

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def match_loss(real_env, virtual_env, input_seed, batch_size):
    states_list = []
    actions_list = []
    outputs_list = []

    real_env.reset()

    for k in range(batch_size):
        # run random state/actions transitions on the real env
        states_list.append(real_env.env.observation_space.sample())
        actions_list.append(real_env.env.action_space.sample())
        next_state, reward, done = real_env.step(
            action=torch.tensor(actions_list[-1], device=device, dtype=torch.float32),
            state=torch.tensor(states_list[-1], device=device, dtype=torch.float32))
        outputs_list.append(torch.cat((next_state, reward.unsqueeze(0), done.unsqueeze(0)), dim=0))

    # convert to torch
    states = torch.tensor(states_list, device=device, dtype=torch.float32)
    actions = torch.tensor(actions_list, device=device, dtype=torch.float32)
    outputs_real = torch.stack(outputs_list)

    # simulate the same state/action transitions on the virtual env
    input_seeds = torch.tensor([input_seed], device=device, dtype=torch.float32).repeat(len(states)).unsqueeze(1)
    next_states_virtual, rewards_virtual, dones_virtual = virtual_env.step(action=actions, state=states,
                                                                           input_seed=input_seeds)
    outputs_virtual = torch.cat([next_states_virtual, rewards_virtual, dones_virtual], dim=1)

    return F.mse_loss(outputs_real, outputs_virtual)


class MatchEnv(nn.Module):
    def __init__(self, config):
        super().__init__()

        me_config = config['agents']['match_env']

        self.lr = me_config['lr']
        self.weight_decay = me_config['weight_decay']
        self.batch_size = me_config['batch_size']
        self.early_out_diff = me_config['early_out_diff']
        self.early_out_num = me_config['early_out_num']
        self.steps = me_config['steps']


    def run(self, real_env, virtual_env, input_seed):
        optimizer = torch.optim.Adam(virtual_env.parameters(),
                                     lr = self.lr,
                                     weight_decay = self.weight_decay)

        avg_meter_loss = AverageMeter(print_str='Average loss')

        old_loss = 0

        for i in range(self.steps):
            # match virtual env to real env
            optimizer.zero_grad()
            loss = match_loss(real_env, virtual_env, input_seed, self.batch_size)
            loss.backward()
            optimizer.step()

            # logging
            avg_meter_loss.update(loss, print_rate=self.early_out_num)

            # early out
            loss = avg_meter_loss.get_mean(num=self.early_out_num)
            if i % self.early_out_num == 0:
                if abs(old_loss-loss) / loss < self.early_out_diff:
                    print('early out')
                    break
                else:
                    old_loss = loss


