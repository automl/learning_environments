import os
import sys

import torch
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis

from agents.agent_utils import select_agent
from envs.env_factory import EnvFactory

# SAVE_DIR = '/home/nierhoff/master_thesis/learning_environments/results/halfcheetah_compare_reward_envs'
SAVE_DIR = '/home/ferreira/Projects/learning_environments/results/halfcheetah_compare_reward_envs'

LOG_DICT = {}
LOG_DICT['1'] = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_halfcheetah_2021-01-22-13_1'
LOG_DICT['2'] = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_halfcheetah_2021-01-22-17_2'
LOG_DICT['3'] = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_halfcheetah_2021-01-24-11_3'
LOG_DICT['4'] = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_halfcheetah_2021-01-24-11_4'
LOG_DICT['5'] = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_halfcheetah_2021-01-22-17_5'
LOG_DICT['6'] = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_halfcheetah_2021-01-22-17_6'
LOG_DICT['7'] = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_halfcheetah_2021-01-24-19_7'
LOG_DICT['8'] = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_halfcheetah_2021-01-24-19_8'
LOG_DICT['101'] = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_halfcheetah_2021-01-29-21_101'
LOG_DICT['102'] = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_halfcheetah_2021-01-28-22_102'

MODEL_NUM = 5
MODEL_AGENTS = 5

def get_best_models_from_log(log_dir):
    if not os.path.isdir(log_dir):
        log_dir = log_dir.replace('nierhoff', 'dingsda')

    result = hpres.logged_results_to_HBS_result(log_dir)

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

    # before AUC (objective minimized)
    # print("sorting from low to high values (non-AUC)")
    # best_models.sort(key=lambda x: x[0])

    # AUC (objective maximized)
    print("sorting from high to low values (AUC)")
    best_models.sort(key=lambda x: x[0], reverse=True)
    best_models = best_models[:MODEL_NUM]

    return best_models


def load_envs_and_config(model_file):
    save_dict = torch.load(model_file)

    config = save_dict['config']
    config['device'] = 'cuda'
    config['envs']['HalfCheetah-v3']['solved_reward'] = 100000  # something big enough to prevent early out triggering

    env_factory = EnvFactory(config=config)
    reward_env = env_factory.generate_reward_env()
    print(save_dict['model'])
    reward_env.load_state_dict(save_dict['model'])
    real_env = env_factory.generate_real_env()

    return reward_env, real_env, config


def train_test_agents(mode, env, real_env, config):
    rewards = []
    episode_lengths = []

    # settings for comparability
    config['agents']['td3']['test_episodes'] = 1
    config['agents']['td3']['train_episodes'] = 1000
    config['agents']['td3']['print_rate'] = 100

    config['agents']['td3']['lr'] = 3e-4
    config['agents']['td3']['tau'] = 0.005
    config['agents']['td3']['activation_fn'] = 'relu'
    config['agents']['td3']['same_action_num'] = 1
    config['agents']['td3']['policy_delay'] = 2
    config['agents']['td3']['hidden_size'] = 128
    config['agents']['td3']['hidden_layer'] = 2
    config['agents']['td3']['policy_std_clip'] = 0.5
    config['agents']['td3']['policy_std'] = 0.2
    config['agents']['td3']['action_std'] = 0.1
    config['agents']['td3']['batch_size'] = 256
    config['agents']['td3']['gamma'] = 0.99
    config['agents']['td3']['rb_size'] = 1000000
    config['agents']['td3']['init_episodes'] = 20  # set via time steps in original td3 implementation

    config['agents']['td3']['early_out_num'] = 50
    config['agents']['td3']['early_out_virtual_diff'] = 0.02

    # optimized ICM HPs:
    # config['agents']['icm'] = {}
    # config['agents']['icm']['beta'] = 0.05
    # config['agents']['icm']['eta'] = 0.01
    # config['agents']['icm']['feature_dim'] = 32
    # config['agents']['icm']['hidden_size'] = 128
    # config['agents']['icm']['lr'] = 1e-5

    # optimized ICM HPs (from max. reward run):
    config['agents']['icm'] = {}
    config['agents']['icm']['beta'] = 0.001
    config['agents']['icm']['eta'] = 0.1
    config['agents']['icm']['feature_dim'] = 32
    config['agents']['icm']['hidden_size'] = 128
    config['agents']['icm']['lr'] = 1e-5

    # default ICM HPs:
    # config['agents']['icm'] = {}
    # config['agents']['icm']['beta'] = 0.2
    # config['agents']['icm']['eta'] = 0.5
    # config['agents']['icm']['feature_dim'] = 64
    # config['agents']['icm']['hidden_size'] = 128
    # config['agents']['icm']['lr'] = 1e-4

    for i in range(MODEL_AGENTS):
        if mode == '-1':
            agent = select_agent(config=config, agent_name='td3_icm')
        else:
            agent = select_agent(config=config, agent_name='td3')
        reward, episode_length, _ = agent.train(env=env, test_env=real_env)
        print('reward: ' + str(reward))
        rewards.append(reward)
        episode_lengths.append(episode_length)
    return rewards, episode_lengths


def save_list(mode, config, reward_list, episode_length_list):
    os.makedirs(SAVE_DIR, exist_ok=True)
    # file_name = os.path.join(SAVE_DIR, 'best' + str(mode) + "_bohb_max_reward_run" + '.pt')
    file_name = os.path.join(SAVE_DIR, 'best' + str(mode) + '.pt')
    save_dict = {}
    save_dict['config'] = config
    save_dict['model_num'] = MODEL_NUM
    save_dict['model_agents'] = MODEL_AGENTS
    save_dict['reward_list'] = reward_list
    save_dict['episode_length_list'] = episode_length_list
    torch.save(save_dict, file_name)


def eval_models(mode, log_dir):
    best_models = get_best_models_from_log(log_dir)

    reward_list = []
    episode_length_list = []

    for best_reward, model_file in best_models:
        print('best reward: ' + str(best_reward))
        print('model file: ' + str(model_file))
        reward_env, real_env, config = load_envs_and_config(model_file)
        rewards, episode_lengths = train_test_agents(mode=mode, env=reward_env, real_env=real_env, config=config)
        reward_list += rewards
        episode_length_list += episode_lengths

    save_list(mode, config, reward_list, episode_length_list)


def eval_base(mode, log_dir):
    best_models = get_best_models_from_log(log_dir)

    reward_list = []
    episode_length_list = []

    for i in range(MODEL_NUM):
        reward_env, real_env, config = load_envs_and_config(best_models[0][1])
        rewards, episode_lengths = train_test_agents(mode=mode, env=real_env, real_env=real_env, config=config)
        reward_list += rewards
        episode_length_list += episode_lengths

    save_list(mode, config, reward_list, episode_length_list)


def eval_icm(mode, log_dir):
    best_models = get_best_models_from_log(log_dir)

    reward_list = []
    episode_length_list = []

    for i in range(MODEL_NUM):
        reward_env, real_env, config = load_envs_and_config(best_models[0][1])
        rewards, episode_lengths = train_test_agents(mode=mode, env=real_env, real_env=real_env, config=config)
        reward_list += rewards
        episode_length_list += episode_lengths

    save_list(mode, config, reward_list, episode_length_list)


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
