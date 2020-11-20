import os
import sys
import torch
from envs.env_factory import EnvFactory
from agents.agent_utils import select_agent

MODEL_NUM = 40
MODEL_AGENTS = 10
# local machine
#MODEL_DIR = '/home/dingsda/master_thesis/learning_environments/results/GTNC_evaluate_cartpole_vary_hp_2020-11-17-10/GTN_models_CartPole-v0'
# cluster
MODEL_DIR = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_cartpole_vary_hp_2020-11-17-10/GTN_models_CartPole-v0'



def load_envs_and_config(file_name):
    file_path = os.path.join(MODEL_DIR, file_name)
    save_dict = torch.load(file_path)
    config = save_dict['config']
    config['device'] = 'cuda'
    env_factory = EnvFactory(config=config)
    virtual_env = env_factory.generate_virtual_env()
    virtual_env.load_state_dict(save_dict['model'])
    real_env = env_factory.generate_default_real_env()

    return virtual_env, real_env, config



def get_all_files(with_vary_hp):
    file_list = []
    for file_name in os.listdir(MODEL_DIR):
        if 'CartPole' not in file_name:
            continue

        _, _, config = load_envs_and_config(file_name)
        if config['agents']['ddqn_vary']['vary_hp'] == with_vary_hp:
            file_list.append(file_name)

    # sort file list by random characters/digits -> make randomness deterministic
    file_list = sorted(file_list, key=lambda elem: elem[-9:])
    if len(file_list) < MODEL_NUM:
        raise ValueError("Not enough saved models")

    return file_list[:MODEL_NUM]


def train_test_agents(train_env, test_env, config):
    reward_list = []

    for i in range(MODEL_AGENTS):
        config['agents']['ddqn_vary']['vary_hp'] = True
        config['agents']['ddqn']['print_rate'] = 10
        agent = select_agent(config=config, agent_name='DDQN_vary')
        agent.train(env=train_env)
        reward, _ = agent.test(env=test_env)
        print('reward: ' + str(reward))
        reward_list.append(reward)

    return reward_list


def save_reward_list(mode, config, reward_list):
    file_name = os.path.join(MODEL_DIR, str(mode)+'.pt')
    save_dict = {}
    save_dict['config'] = config
    save_dict['reward_list'] = reward_list
    torch.save(save_dict, file_name)


def run_vary_hp(mode):
    if mode == 0:
        train_on_venv = False
    elif mode == 1:
        train_on_venv = True
        with_vary_hp = False
    elif mode == 2:
        train_on_venv = True
        with_vary_hp = True

    reward_list = []

    if not train_on_venv:
        file_name = os.listdir(MODEL_DIR)[0]
        _, real_env, config = load_envs_and_config(file_name)

        for i in range(MODEL_NUM):
            print('train on {}-th environment'.format(i))
            reward_list += train_test_agents(train_env=real_env, test_env=real_env, config=config)

    else:
        file_list = get_all_files(with_vary_hp=with_vary_hp)

        for file_name in file_list:
            virtual_env, real_env, config = load_envs_and_config(file_name)
            print('train agents on ' + str(file_name))
            reward_list += train_test_agents(train_env=virtual_env, test_env=real_env, config=config)

    save_reward_list(mode=mode, config=config, reward_list=reward_list)



if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_vary_hp(mode=int(int(sys.argv[1])))
    else:
        for mode in range(3):
            run_vary_hp(mode=mode)