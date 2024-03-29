import copy

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import torch
import yaml

from agents.TD3 import TD3
from envs.env_factory import EnvFactory


class TD3_vary(TD3):
    def __init__(self, env, max_action, config, icm=False):
        self.agent_name = 'td3'

        if config["agents"]["td3_vary"]["vary_hp"]:
            config_mod = copy.deepcopy(config)
            config_mod = self.vary_hyperparameters(config_mod)
        else:
            config_mod = config

        super().__init__(env=env, max_action=max_action, config=config_mod, icm=icm)

    def vary_hyperparameters(self, config_mod):

        lr = config_mod['agents'][self.agent_name]['lr']
        batch_size = config_mod['agents'][self.agent_name]['batch_size']
        hidden_size = config_mod['agents'][self.agent_name]['hidden_size']
        hidden_layer = config_mod['agents'][self.agent_name]['hidden_layer']

        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='lr', lower=lr / 3, upper=lr * 3, log=True, default_value=lr))
        cs.add_hyperparameter(
            CSH.UniformIntegerHyperparameter(name='batch_size', lower=int(batch_size / 3), upper=int(batch_size * 3), log=True,
                                             default_value=batch_size))
        cs.add_hyperparameter(
            CSH.UniformIntegerHyperparameter(name='hidden_size', lower=int(hidden_size / 3), upper=int(hidden_size * 3), log=True,
                                             default_value=hidden_size))
        cs.add_hyperparameter(
            CSH.UniformIntegerHyperparameter(name='hidden_layer', lower=hidden_layer - 1, upper=hidden_layer + 1, log=False,
                                             default_value=hidden_layer))

        config = cs.sample_configuration()

        print(f"sampled part of config: "
              f"lr: {config['lr']}, "
              f"batch_size: {config['batch_size']}, "
              f"hidden_size: {config['hidden_size']}, "
              f"hidden_layer: {config['hidden_layer']}"
              )

        config_mod['agents'][self.agent_name]['lr'] = config['lr']
        config_mod['agents'][self.agent_name]['batch_size'] = config['batch_size']
        config_mod['agents'][self.agent_name]['hidden_size'] = config['hidden_size']
        config_mod['agents'][self.agent_name]['hidden_layer'] = config['hidden_layer']

        print("full config: ", config_mod['agents'][self.agent_name])

        return config_mod


if __name__ == "__main__":
    with open("../default_config_halfcheetah_reward_env.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    torch.set_num_threads(1)

    # generate environment
    env_fac = EnvFactory(config)
    virt_env = env_fac.generate_virtual_env()
    real_env = env_fac.generate_real_env()

    timing = []
    for i in range(10):
        td3 = TD3_vary(env=real_env, max_action=real_env.get_max_action(), config=config)
        # ddqn.train(env=virt_env, time_remaining=50)
        print('TRAIN')
        td3.train(env=real_env, time_remaining=500)
        # print('TEST')
        # ddqn.test(env=real_env, time_remaining=500)
    print('avg. ' + str(sum(timing) / len(timing)))
