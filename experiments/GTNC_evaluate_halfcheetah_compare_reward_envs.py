import os
import sys

import torch
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis

from agents.agent_utils import select_agent
from envs.env_factory import EnvFactory
from models.baselines import ICMTD3

SAVE_DIR = '/home/nierhoff/master_thesis/learning_environments/results/halfcheetah_compare_reward_envs'

LOG_DICT = {}
LOG_DICT['2'] = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_halfcheetah_2021-01-11-19_2'
LOG_DICT['4'] = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_halfcheetah_2021-01-11-19_4'
LOG_DICT['102']= '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_halfcheetah_2021-01-12-11_102'

MODEL_NUM = 5
MODEL_AGENTS = 3

def get_best_models_from_log(log_dir):
    result = hpres.logged_results_to_HBS_result(log_dir)
    all_runs = result.get_all_runs()
    id2conf = result.get_id2config_mapping()

    best_models = []

    for value in result.data.values():
        try:
            loss = value.results[1.0]['loss']
            model_name = value.results[1.0]['info']['model_name']

            if not os.path.isfile(model_name):
                model_name = model_name.replace('nierhoff', 'dingsda')
            best_models.append((loss, model_name))
        except:
            continue

    best_models.sort(key=lambda x: x[0], reverse=True)
    best_models = best_models[:MODEL_NUM]

    return best_models


def load_envs_and_config(model_file):
    file_path = os.path.join(model_file)
    save_dict = torch.load(file_path)

    config = save_dict['config']
    config['device'] = 'cuda'
    config['envs']['HalfCheetah-v3']['solved_reward'] = 100000  # something big enough to prevent early out triggering

    env_factory = EnvFactory(config=config)
    reward_env = env_factory.generate_reward_env()
    reward_env.load_state_dict(save_dict['model'])
    real_env = env_factory.generate_real_env()

    return reward_env, real_env, config


def train_test_agents(mode, env, real_env, config):
    rewards = []

    # settings for comparability
    config['agents']['td3']['test_episodes'] = 1
    config['agents']['td3']['train_episodes'] = 100
    config['agents']['td3']['print_rate'] = 1

    for i in range(MODEL_AGENTS):
        if mode == '-1':
            agent = ICMTD3(env=real_env, max_action=real_env.get_max_action(), config=config)
        else:
            agent = select_agent(config=config, agent_name='td3')
        reward, _ = agent.train(env=env, real_env=real_env, test_env=real_env)
        print('reward: ' + str(reward))
        rewards.append(reward)

    return rewards


def save_list(mode, config, reward_list):

    os.makedirs(SAVE_DIR, exist_ok=True)
    file_name = os.path.join(SAVE_DIR, 'worst' + str(MODEL_NUM) + '_' + str(mode) + '.pt')
    save_dict = {}
    save_dict['config'] = config
    save_dict['reward_list'] = reward_list
    print(reward_list)
    torch.save(save_dict, file_name)


def eval_models(mode, log_dir):
    best_models = get_best_models_from_log(log_dir)

    reward_list = []
    for _, model_file in best_models:
        reward_env, real_env, config = load_envs_and_config(model_file)
        rewards = train_test_agents(mode=mode, env=reward_env, real_env=real_env, config=config)
        reward_list += rewards
    save_list(mode, config, reward_list)


def eval_base(mode, log_dir):
    best_models = get_best_models_from_log(log_dir)

    reward_list = []
    for i in range(MODEL_NUM):
        reward_env, real_env, config = load_envs_and_config(best_models[0][1])
        rewards = train_test_agents(mode=mode, env=real_env, real_env=real_env, config=config)
        reward_list += rewards
    save_list(mode, config, reward_list)


def eval_icm(mode, log_dir):
    best_models = get_best_models_from_log(log_dir)

    reward_list = []
    for i in range(MODEL_NUM):
        reward_env, real_env, config = load_envs_and_config(best_models[0][1])
        rewards = train_test_agents(mode=mode, env=real_env, real_env=real_env, config=config)
        reward_list += rewards
    save_list(mode, config, reward_list)


if __name__ == "__main__":

    for arg in sys.argv[1:]:
        print(arg)
    mode = str(sys.argv[1])

    if mode == '-1':
        eval_icm(mode=mode, log_dir=LOG_DICT['2'])
    elif mode == '0':
        eval_base(mode=mode, log_dir=LOG_DICT['2'])
    else:
        eval_models(mode=mode, log_dir=LOG_DICT[mode])

