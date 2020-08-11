import yaml
import math
import torch
import torch.nn as nn
from envs.env_factory import EnvFactory
from utils import AverageMeter, ReplayBuffer, print_abs_param_sum, print_avg_pairwise_dist

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




class EnvMatcher(nn.Module):
    def __init__(self, config):
        super().__init__()

        em_config = config["agents"]["env_matcher"]

        self.lr = em_config["lr"]
        self.weight_decay = em_config["weight_decay"]
        self.batch_size = em_config["batch_size"]
        self.early_out_diff = em_config["early_out_diff"]
        self.early_out_num = em_config["early_out_num"]
        self.max_steps = em_config["max_steps"]
        self.step_size = em_config["step_size"]
        self.gamma = em_config["gamma"]
        self.rb_use = em_config["rb_use"]
        self.match_loss_state = em_config["match_loss_state"]
        self.match_loss_reward = em_config["match_loss_reward"]
        self.match_loss_done = em_config["match_loss_done"]
        self.variation_type = em_config["variation_type"]
        self.variation_weight = em_config["variation_weight"]

        self.step = 0

    def train(self, virtual_env, input_seeds, replay_buffer):
        optimizer = torch.optim.Adam(list(virtual_env.parameters()) + input_seeds, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        avg_meter_loss =       AverageMeter(print_str="Average loss       ")
        avg_meter_match_loss = AverageMeter(print_str="Average match loss ")
        avg_meter_var_loss =   AverageMeter(print_str="Average var loss   ")
        avg_meter_state =      AverageMeter(print_str="Average diff state    ")
        avg_meter_reward =     AverageMeter(print_str="Average diff reward   ")
        avg_meter_done =       AverageMeter(print_str="Average diff done     ")
        avg_meter_rb_size =    AverageMeter(print_str="Average rb size          ")

        old_loss = float('Inf')

        for self.step in range(self.max_steps):
            # match virtual env to real env
            optimizer.zero_grad()
            loss, match_loss, var_loss, diff_state, diff_reward, diff_done = \
                self.match_loss(virtual_env=virtual_env,
                                input_seeds=input_seeds,
                                replay_buffer=replay_buffer,
                                batch_size=self.batch_size,
                                more_info=True,
                                grad_enabled=True)

            loss.backward()
            optimizer.step()
            scheduler.step()

            # logging
            avg_meter_loss.update(loss, print_rate=self.early_out_num)
            avg_meter_match_loss.update(match_loss, print_rate=self.early_out_num)
            avg_meter_var_loss.update(var_loss, print_rate=self.early_out_num)
            avg_meter_state.update(diff_state, print_rate=self.early_out_num)
            avg_meter_reward.update(diff_reward, print_rate=self.early_out_num)
            avg_meter_done.update(diff_done, print_rate=self.early_out_num)
            avg_meter_rb_size.update(replay_buffer.get_size(), print_rate=self.early_out_num)

            # early out
            loss = avg_meter_loss.get_mean(num=self.early_out_num)
            if self.step % self.early_out_num == 0:
                #print_abs_param_sum(model=virtual_env, name='virtual_env: ')
                print(input_seeds)
                print_avg_pairwise_dist(vec_list=input_seeds, name='avg. input seed distance')
                if abs(old_loss - loss) / loss < self.early_out_diff:
                    print("early out")
                    break
                else:
                    old_loss = loss

        return avg_meter_loss.get_raw_data()


    def test(self, virtual_env, input_seeds, replay_buffer, test_samples):
        loss, diff_state, diff_reward, diff_done = \
            self.match_loss(virtual_env=virtual_env,
                            input_seeds=input_seeds,
                            replay_buffer=replay_buffer,
                            batch_size=test_samples,
                            more_info=True,
                            grad_enabled=False)

        return loss, diff_state, diff_reward, diff_done


    def match_loss(self, virtual_env, input_seeds, replay_buffer, batch_size, more_info=False, grad_enabled=True):
        with torch.set_grad_enabled(grad_enabled):
            batch_size_normalized = batch_size // len(input_seeds) + 1
            states, actions, next_states, rewards, dones = replay_buffer.sample(batch_size_normalized)
            outputs_real = torch.cat((next_states, rewards, dones), dim=1).to(device)

            outputs_virtual_list = []
            for input_seed in input_seeds:
                input_seed_rep = input_seed.repeat(len(states), 1)
                next_states_virtual, rewards_virtual, dones_virtual = virtual_env.step(action=actions,
                                                                                       state=states,
                                                                                       input_seed=input_seed_rep)
                outputs_virtual_list.append(torch.cat([next_states_virtual, rewards_virtual, dones_virtual], dim=1).to(device))

            # todo fabio: maybe make loss as parameter (low priority)
            loss = 0
            state_loss_fct = self.get_loss_function(self.match_loss_state)
            reward_loss_fct = self.get_loss_function(self.match_loss_reward)
            done_loss_fct = self.get_loss_function(self.match_loss_done)

            match_loss = 0
            for i in range(len(input_seeds)):
                match_loss += state_loss_fct(outputs_real[:, :-2], outputs_virtual_list[i][:, :-2])
                match_loss += reward_loss_fct(outputs_real[:, -2], outputs_virtual_list[i][:, -2])
                match_loss += done_loss_fct(outputs_real[:, -1], outputs_virtual_list[i][:, -1])
            loss += match_loss

            variation_loss = 0
            if self.variation_weight > 0:
                for i in range(len(input_seeds)):
                    for k in range(i+1, len(input_seeds)):
                        if self.variation_type == 1:
                            variation_loss += torch.mean(torch.abs(outputs_virtual_list[i][:, :-2]-outputs_virtual_list[k][:, :-2]))
                        elif self.variation_type == 2:
                            variation_loss += torch.mean(torch.abs(outputs_virtual_list[i][:, :-1]-outputs_virtual_list[k][:, :-1]))
                        elif self.variation_type == 3:
                            variation_loss += torch.mean(torch.abs(outputs_virtual_list[i]-outputs_virtual_list[k]))
                        else:
                            raise ValueError("Unknown variation_type value: " + str(self.variation_type))

                if len(input_seeds) > 1:
                    variation_loss /= len(input_seeds)
                    variation_loss = -torch.log(variation_loss) * self.variation_weight
                    loss += variation_loss

            if more_info:
                avg_diff_state, avg_diff_reward, avg_diff_done = self.get_log_information(outputs_real=outputs_real,
                                                                                          outputs_virtual_list=outputs_virtual_list)
                return loss, match_loss, variation_loss, avg_diff_state, avg_diff_reward, avg_diff_done
            else:
                return loss


    def get_loss_function(self, name):
        if name == 'L1':
            return torch.nn.L1Loss()
        elif name == 'L2':
            return torch.nn.MSELoss()
        else:
            raise NotImplementedError("Unknown loss function: " + str(name))


    def get_log_information(self, outputs_real, outputs_virtual_list):
        n = len(outputs_virtual_list)
        avg_diff_state = torch.zeros(outputs_real[:, :-2].shape)
        avg_diff_reward = torch.zeros(outputs_real[:, -2].shape)
        avg_diff_done = torch.zeros(outputs_real[:, -1].shape)
        for i in range(n):
            avg_diff_state += abs(outputs_real[:, :-2].cpu() - outputs_virtual_list[i][:, :-2].cpu()) / n
            avg_diff_reward += abs(outputs_real[:, -2].cpu() - outputs_virtual_list[i][:, -2].cpu()) / n
            avg_diff_done += abs(outputs_real[:, -1].cpu() - outputs_virtual_list[i][:, -1].cpu()) / n

        return avg_diff_state.mean(), avg_diff_reward.mean(), avg_diff_done.mean()


if __name__ == "__main__":
    with open("../default_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    # generate environments
    env_fac = EnvFactory(config)
    virtual_env = env_fac.generate_default_virtual_env()

    real_env = env_fac.generate_random_real_env()
    input_seeds = []
    for i in range(10):
        input_seeds.append(env_fac.generate_random_input_seed())

    me = EnvMatcher(config)
    me.train(real_env=real_env, virtual_env=virtual_env, input_seeds=input_seeds)
