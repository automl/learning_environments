import yaml
import torch
import time
import copy
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from agents.DDQN import DDQN
from envs.env_factory import EnvFactory


class DDQN_vary(DDQN):
    def __init__(self, env, config):
        self.agent_name = 'ddqn'

        if config["agents"]["ddqn_vary"]["vary_hp"]:
            config_mod = copy.deepcopy(config)
            config_mod = self.vary_heyperparameters(config_mod)
        else:
            config_mod = config

        print(config_mod['agents'][self.agent_name]['lr'])
        print(config_mod['agents'][self.agent_name]['hidden_size'])
        print(config_mod['agents'][self.agent_name]['hidden_layer'])

        super().__init__(env=env, config=config_mod)


    def vary_heyperparameters(self, config_mod):

        lr = config_mod['agents'][self.agent_name]['lr']
        hidden_size = config_mod['agents'][self.agent_name]['hidden_size']
        hidden_layer = config_mod['agents'][self.agent_name]['hidden_layer']

        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='lr', lower=lr/3, upper=lr*3, log=True, default_value=lr))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='hidden_size', lower=int(hidden_size/3), upper=int(hidden_size*3), log=True, default_value=hidden_size))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='hidden_layer', lower=hidden_layer-1, upper=hidden_layer+1, log=False, default_value=hidden_layer))

        config = cs.sample_configuration()

        print(config_mod['agents'][self.agent_name])
        config_mod['agents'][self.agent_name]['lr'] = config['lr']
        config_mod['agents'][self.agent_name]['hidden_size'] = config['hidden_size']
        config_mod['agents'][self.agent_name]['hidden_layer'] = config['hidden_layer']

        return config_mod


if __name__ == "__main__":
    with open("../default_config_cartpole.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    torch.set_num_threads(1)

    # generate environment
    env_fac = EnvFactory(config)
    virt_env = env_fac.generate_virtual_env()
    real_env = env_fac.generate_default_real_env()

    timing = []
    for i in range(10):
        ddqn = DDQN_vary(env=real_env, config=config)

        #ddqn.train(env=virt_env, time_remaining=50)

        t1 = time.time()
        print('TRAIN')
        ddqn.train(env=real_env, time_remaining=500)
        t2 = time.time()
        timing.append(t2-t1)
        print(t2-t1)
        #print('TEST')
        #reward_list = ddqn.test(env=real_env, time_remaining=500)
    print('avg. ' + str(sum(timing)/len(timing)))

