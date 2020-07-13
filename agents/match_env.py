import yaml
import torch
import torch.nn as nn
from envs.env_factory import EnvFactory
from utils import AverageMeter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def match_loss(real_env, virtual_env, input_seed, batch_size, more_info=False, grad_enabled=True, oversampling=1.1):
    with torch.set_grad_enabled(grad_enabled):
        states_list = []
        actions_list = []
        outputs_list = []

        real_env.reset()

        for k in range(batch_size):
            # run random state/actions transitions on the real env
            states_list.append(real_env.env.observation_space.sample()*oversampling) # the 1.1 is important!
            actions_list.append(real_env.env.action_space.sample()*oversampling)     # the 1.1 is important!
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
        next_states_virtual, rewards_virtual, dones_virtual = virtual_env.step(action=actions,
                                                                               state=states,
                                                                               input_seed=input_seeds)
        outputs_virtual = torch.cat([next_states_virtual, rewards_virtual, dones_virtual], dim=1)
        #
        # print('----')
        # print(outputs_real[0,:])
        # print(outputs_virtual[0,:])

        loss_fkt = torch.nn.L1Loss()
        avg_loss = loss_fkt(outputs_real, outputs_virtual).to(device)

        avg_diff_state   = abs(outputs_real[:,:-2].cpu() - outputs_virtual[:,:-2].cpu()).sum() / batch_size
        avg_diff_reward  = abs(outputs_real[:,-2].cpu()  - outputs_virtual[:,-2].cpu()).sum()  / batch_size
        avg_diff_done    = abs(outputs_real[:,-1].cpu()  - outputs_virtual[:,-1].cpu()).sum()  / batch_size

        if more_info:
            return avg_loss, avg_diff_state, avg_diff_reward, avg_diff_done
        else:
            return avg_loss


class MatchEnv(nn.Module):
    def __init__(self, config):
        super().__init__()

        me_config = config['agents']['match_env']

        self.oversampling = me_config['oversampling']
        self.lr = me_config['lr']
        self.weight_decay = me_config['weight_decay']
        self.batch_size = me_config['batch_size']
        self.early_out_diff = me_config['early_out_diff']
        self.early_out_num = me_config['early_out_num']
        self.steps = me_config['steps']
        self.step_size = me_config['step_size']
        self.gamma = me_config['gamma']

    def train(self, real_env, virtual_env, input_seed):
        optimizer = torch.optim.Adam(virtual_env.parameters(),
                                     lr = self.lr,
                                     weight_decay = self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=self.step_size,
                                                    gamma=self.gamma)
        avg_meter_loss   = AverageMeter(print_str='Average loss')
        avg_meter_state  = AverageMeter(print_str='Average diff state  ')
        avg_meter_reward = AverageMeter(print_str='Average diff reward ')
        avg_meter_done   = AverageMeter(print_str='Average diff done   ')

        old_loss = 0

        for i in range(self.steps):
            # match virtual env to real env
            optimizer.zero_grad()
            loss, diff_state, diff_reward, diff_done = \
                match_loss(real_env = real_env,
                           virtual_env = virtual_env,
                           input_seed = input_seed,
                           batch_size = self.batch_size,
                           grad_enabled = True,
                           more_info = True,
                           oversampling = self.oversampling)

            loss.backward()
            optimizer.step()
            scheduler.step()

            # logging
            avg_meter_loss.update(loss, print_rate=self.early_out_num)
            avg_meter_state.update(diff_state, print_rate=self.early_out_num)
            avg_meter_reward.update(diff_reward, print_rate=self.early_out_num)
            avg_meter_done.update(diff_done, print_rate=self.early_out_num)

            # early out
            loss = avg_meter_loss.get_mean(num=self.early_out_num)
            if i % self.early_out_num == 0:
                #optimizer = self.init_optimizer(virtual_env)
                if abs(old_loss-loss) / loss < self.early_out_diff:
                    print('early out')
                    break
                else:
                    old_loss = loss

        return avg_meter_loss.get_raw_data()


    def test(self, real_env, virtual_env, input_seed, oversampling, test_samples):
        loss, diff_state, diff_reward, diff_done = \
            match_loss(real_env=real_env,
                       virtual_env=virtual_env,
                       input_seed=input_seed,
                       batch_size=test_samples,
                       grad_enabled=False,
                       more_info=True,
                       oversampling=oversampling)
        return loss, diff_state, diff_reward, diff_done


if __name__ == "__main__":
    with open("../default_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    # generate environment
    env_fac = EnvFactory(config)
    real_env = env_fac.generate_default_real_env()
    virtual_env = env_fac.generate_default_virtual_env()

    me = MatchEnv(config)
    me.train(real_env=real_env, virtual_env=virtual_env, input_seed=0)



