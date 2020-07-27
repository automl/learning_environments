import yaml
import math
import torch
import torch.nn as nn
from envs.env_factory import EnvFactory
from utils import AverageMeter, ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def match_loss(real_env, virtual_env, input_seed, batch_size, multi_step=1, more_info=False, grad_enabled=True, oversampling=1.1, replay_buffer=None):
    with torch.set_grad_enabled(grad_enabled):

        real_env.reset()

        # do not use replay buffer
        if replay_buffer == None:
            actions_list = []
            states_list = []
            outputs_list = []

            for k in range(batch_size):
                # run random state/actions transitions on the real env
                # todo fabio: maybe improve (access member variables)
                actions_list.append(real_env.get_random_action() * oversampling)
                states_list.append(real_env.get_random_state() * oversampling)

                next_state_tmp = states_list[-1]
                next_state_list = []
                for i in range(multi_step):
                    next_state_tmp, reward, done = real_env.step(action = actions_list[-1],
                                                                 state = next_state_tmp)
                    next_state_list.append(next_state_tmp)
                next_state = torch.cat(next_state_list, dim=0)
                outputs_list.append(torch.cat((next_state, reward.unsqueeze(0), done.unsqueeze(0)), dim=0))

            # convert to torch
            actions = torch.stack(actions_list).to(device)
            states = torch.stack(states_list).to(device)
            outputs_real = torch.stack(outputs_list).to(device)

        else:
            # first fill replay buffer with enough values
            if replay_buffer.get_size() < batch_size:
                num_samples = int(batch_size*10)
            else:
                num_samples = int(batch_size/10)+1

            dummy = torch.tensor([0])
            for k in range(num_samples):
                action = real_env.get_random_action() * oversampling
                state = real_env.get_random_state() * oversampling

                next_state_tmp = state
                next_state_list = []
                for i in range(multi_step):
                    next_state_tmp, reward, done = real_env.step(action = action,
                                                                 state = next_state_tmp)
                    next_state_list.append(next_state_tmp)
                next_state = torch.cat(next_state_list, dim=0)
                replay_buffer.add(last_state=dummy, last_action=dummy, state=state, action=action, next_state=next_state, reward=reward, done=done)

            _, _, states, actions, next_states, rewards, dones = replay_buffer.sample(batch_size)
            outputs_real = torch.cat((next_states, rewards, dones), dim=1).to(device)

        # simulate the same state/action transitions on the virtual env, create input_seeds batch
        input_seeds = input_seed.repeat(len(states), 1)

        next_states_virtual_tmp = states
        next_states_virtual_list = []
        for i in range(multi_step):
            next_states_virtual_tmp, rewards_virtual, dones_virtual = virtual_env.step(action=actions, state=next_states_virtual_tmp, input_seed=input_seeds)
            next_states_virtual_list.append(next_states_virtual_tmp)
        next_states_virtual = torch.cat(next_states_virtual_list, dim=1)
        outputs_virtual = torch.cat([next_states_virtual, rewards_virtual, dones_virtual], dim=1).to(device)

        # todo fabio: maybe make loss as parameter (low priority)
        loss_fkt = torch.nn.L1Loss()
        avg_loss = loss_fkt(outputs_real, outputs_virtual)

        avg_diff_state = abs(outputs_real[:, :-2].cpu() - outputs_virtual[:, :-2].cpu()).sum() / batch_size
        avg_diff_reward = abs(outputs_real[:, -2].cpu() - outputs_virtual[:, -2].cpu()).sum() / batch_size
        avg_diff_done = abs(outputs_real[:, -1].cpu() - outputs_virtual[:, -1].cpu()).sum() / batch_size

        if more_info:
            return avg_loss, avg_diff_state, avg_diff_reward, avg_diff_done
        else:
            return avg_loss


class EnvMatcher(nn.Module):
    def __init__(self, config):
        super().__init__()

        em_config = config["agents"]["env_matcher"]

        self.oversampling = em_config["oversampling"]
        self.lr = em_config["lr"]
        self.weight_decay = em_config["weight_decay"]
        self.batch_size = em_config["batch_size"]
        self.early_out_diff = em_config["early_out_diff"]
        self.early_out_num = em_config["early_out_num"]
        self.max_steps = em_config["max_steps"]
        self.step_size = em_config["step_size"]
        self.gamma = em_config["gamma"]
        self.multi_step = em_config["multi_step"]
        self.use_rb = em_config["use_rb"]

    def train(self, real_envs, virtual_env, input_seeds):
        optimizer = torch.optim.Adam(list(virtual_env.parameters()) + input_seeds, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        avg_meter_loss = AverageMeter(print_str="Average loss")
        avg_meter_state = AverageMeter(print_str="Average diff state  ")
        avg_meter_reward = AverageMeter(print_str="Average diff reward ")
        avg_meter_done = AverageMeter(print_str="Average diff done   ")

        old_loss = float('Inf')
        n = len(real_envs)
        batch_size_normalized = math.ceil(self.batch_size / n)

        # initialize replay buffers
        replay_buffers = []
        for k in range(len(real_envs)):
            if self.use_rb:
                real_env = real_envs[k]
                replay_buffers.append(ReplayBuffer(state_dim=real_env.get_state_dim(),
                                                   action_dim=real_env.get_action_dim(),
                                                   max_size=int(1e6)))
            else:
                replay_buffers.append(None)

        for i in range(self.max_steps):
            # match virtual env to real env
            loss, diff_state, diff_reward, diff_done = 0, 0, 0, 0

            optimizer.zero_grad()
            for real_env, input_seed, replay_buffer in zip(real_envs, input_seeds, replay_buffers):
                loss_tmp, diff_state_tmp, diff_reward_tmp, diff_done_tmp = \
                    match_loss(real_env=real_env,
                               virtual_env=virtual_env,
                               input_seed=input_seed,
                               batch_size=batch_size_normalized,
                               multi_step=self.multi_step,
                               more_info=True,
                               grad_enabled=True,
                               oversampling=self.oversampling,
                               replay_buffer=replay_buffer)
                loss += loss_tmp
                diff_state += diff_state_tmp
                diff_reward += diff_reward_tmp
                diff_done += diff_done_tmp

            loss.backward()
            optimizer.step()
            scheduler.step()

            # logging
            avg_meter_loss.update(loss / n, print_rate=self.early_out_num)
            avg_meter_state.update(diff_state / n, print_rate=self.early_out_num)
            avg_meter_reward.update(diff_reward / n, print_rate=self.early_out_num)
            avg_meter_done.update(diff_done / n, print_rate=self.early_out_num)

            # early out
            loss = avg_meter_loss.get_mean(num=self.early_out_num)
            if i % self.early_out_num == 0:
                if abs(old_loss - loss) / loss < self.early_out_diff:
                    print("early out")
                    break
                else:
                    old_loss = loss

        return avg_meter_loss.get_raw_data()

    def test(self, real_envs, virtual_env, input_seeds, oversampling, test_samples):
        loss, diff_state, diff_reward, diff_done = 0, 0, 0, 0

        n = len(real_envs)
        test_samples_normalized = math.ceil(test_samples / n)
        for real_env, input_seed in zip(real_envs, input_seeds):
            loss_tmp, diff_state_tmp, diff_reward_tmp, diff_done_tmp = \
                match_loss(real_env=real_env,
                           virtual_env=virtual_env,
                           input_seed=input_seed,
                           batch_size=test_samples_normalized,
                           more_info=True,
                           grad_enabled=False,
                           oversampling=oversampling)
            loss += loss_tmp
            diff_state += diff_state_tmp
            diff_reward += diff_reward_tmp
            diff_done += diff_done_tmp

        return loss / n, diff_state / n, diff_reward / n, diff_done / n


if __name__ == "__main__":
    with open("../default_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    # generate environments
    env_fac = EnvFactory(config)
    virtual_env = env_fac.generate_default_virtual_env()

    real_envs = []
    input_seeds = []
    for i in range(10):
        real_envs.append(env_fac.generate_random_real_env())
        input_seeds.append(env_fac.generate_random_input_seed())

    me = EnvMatcher(config)
    me.train(real_envs=real_envs, virtual_env=virtual_env, input_seeds=input_seeds)
