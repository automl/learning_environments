import yaml
import torch
import torch.nn as nn
from envs.env_factory import EnvFactory
from agents.agent_utils import select_agent
from utils import AverageMeter

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
        self.match_loss_state = em_config["match_loss_state"]
        self.match_loss_reward = em_config["match_loss_reward"]
        self.match_loss_done = em_config["match_loss_done"]

        self.step = 0

    def train(self, virtual_env, replay_buffer):
        optimizer = torch.optim.Adam(virtual_env.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        avg_meter_loss =       AverageMeter(print_str="Average loss       ")
        avg_meter_state =      AverageMeter(print_str="Average diff state    ")
        avg_meter_reward =     AverageMeter(print_str="Average diff reward   ")
        avg_meter_done =       AverageMeter(print_str="Average diff done     ")
        avg_meter_rb_size =    AverageMeter(print_str="Average rb size          ")

        old_loss = float('Inf')

        for self.step in range(self.max_steps):
            # match virtual env to real env
            optimizer.zero_grad()
            loss, diff_state, diff_reward, diff_done = \
                self.match_loss(virtual_env=virtual_env,
                                replay_buffer=replay_buffer,
                                batch_size=self.batch_size,
                                more_info=True,
                                grad_enabled=True)

            loss.backward()
            optimizer.step()
            scheduler.step()

            # logging
            avg_meter_loss.update(loss, print_rate=self.early_out_num)
            avg_meter_state.update(diff_state, print_rate=self.early_out_num)
            avg_meter_reward.update(diff_reward, print_rate=self.early_out_num)
            avg_meter_done.update(diff_done, print_rate=self.early_out_num)
            avg_meter_rb_size.update(replay_buffer.get_size(), print_rate=self.early_out_num)

            # early out
            loss = avg_meter_loss.get_mean(num=self.early_out_num)
            if self.step % self.early_out_num == 0:
                #print_abs_param_sum(model=virtual_env, name='virtual_env: ')
                if abs(old_loss - loss) / loss < self.early_out_diff:
                    print("early out")
                    break
                else:
                    old_loss = loss

        return avg_meter_loss.get_raw_data()


    def test(self, virtual_env, replay_buffer, test_samples):
        loss, diff_state, diff_reward, diff_done = \
            self.match_loss(virtual_env=virtual_env,
                            replay_buffer=replay_buffer,
                            batch_size=test_samples,
                            more_info=True,
                            grad_enabled=False)

        return loss, diff_state, diff_reward, diff_done


    def match_loss(self, virtual_env, replay_buffer, batch_size, more_info=False, grad_enabled=True):
        with torch.set_grad_enabled(grad_enabled):
            states, actions, next_states, rewards, dones = replay_buffer.sample(batch_size)
            next_states_virtual, rewards_virtual, dones_virtual = virtual_env.env.step(state=states,
                                                                                       action=actions)

            loss = 0
            state_loss_fct = self.get_loss_function(self.match_loss_state)
            reward_loss_fct = self.get_loss_function(self.match_loss_reward)
            done_loss_fct = self.get_loss_function(self.match_loss_done)

            loss += state_loss_fct(next_states, next_states_virtual)
            loss += reward_loss_fct(rewards, rewards_virtual)
            loss += done_loss_fct(dones, dones_virtual)

            avg_diff_state = abs(next_states.cpu() - next_states_virtual.cpu()).mean()
            avg_diff_reward = abs(rewards.cpu() - rewards_virtual.cpu()).mean()
            avg_diff_done = abs(dones.cpu() - dones_virtual.cpu()).mean()

            if more_info:
                return loss, avg_diff_state, avg_diff_reward, avg_diff_done
            else:
                return loss


    def get_loss_function(self, name):
        if name == 'L1':
            return torch.nn.L1Loss()
        elif name == 'L2':
            return torch.nn.MSELoss()
        else:
            raise NotImplementedError("Unknown loss function: " + str(name))


if __name__ == "__main__":
    with open("../default_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    # generate environments
    env_fac = EnvFactory(config)

    virtual_env = env_fac.generate_virtual_env()
    real_env = env_fac.generate_random_real_env()


    agent = select_agent(config, 'TD3')
    print('-- fill replay buffer --')
    reward_list, replay_buffer = agent.train(env=real_env)
    me = EnvMatcher(config)
    print('-- match --')
    me.train(virtual_env=virtual_env, replay_buffer=replay_buffer)

    agent = select_agent(config, 'TD3')
    print('-- train on virtual env --')
    agent.train(env=virtual_env)
    print(' -- train on real env --')
    #agent.reset_optimizer()
    agent.train(env=real_env)