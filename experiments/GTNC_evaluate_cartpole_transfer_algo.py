import os
import sys

import hpbandster.core.result as hpres
import torch

from agents.agent_utils import select_agent
from envs.env_factory import EnvFactory

# SAVE_DIR = '/home/nierhoff/master_thesis/learning_environments/results/cartpole_compare_reward_envs'
# SAVE_DIR = '/home/ferreira/Projects/learning_environments/results/cartpole_compare_reward_envs_5000_train_eps'
# SAVE_DIR = '/home/ferreira/Projects/learning_environments/results/cartpole_compare_reward_envs'
# SAVE_DIR = '/home/ferreira/Projects/learning_environments/results/3_rn_auc/cartpole_compare_reward_envs'
SAVE_DIR = '/home/ferreira/Projects/learning_environments/results/4_rn_reward/cartpole_compare_reward_envs'

LOG_DICT = {}
# LOG_DICT['1'] = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_cartpole_2021-01-28-18_1'
# LOG_DICT['2'] = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_cartpole_2021-01-28-18_2'
# LOG_DICT['5'] = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_cartpole_2021-01-28-18_5'
# LOG_DICT['6'] = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_cartpole_2021-01-28-18_6'

# AUC objective
# LOG_DICT['1'] = '/home/ferreira/Projects/learning_environments/results/GTNC_evaluate_auc_cartpole_2021-05-18-11_1'
# LOG_DICT['2'] = '/home/ferreira/Projects/learning_environments/results/GTNC_evaluate_auc_cartpole_2021-05-15-17_2'
# LOG_DICT['5'] = '/home/ferreira/Projects/learning_environments/results/GTNC_evaluate_auc_cartpole_2021-05-19-20_5'
# LOG_DICT['6'] = '/home/ferreira/Projects/learning_environments/results/GTNC_evaluate_auc_cartpole_2021-05-23-11_6'

LOG_DICT['1'] = '/home/ferreira/Projects/learning_environments/results/GTNC_evaluate_reward_cartpole_2021-06-01-19_1'
LOG_DICT['2'] = '/home/ferreira/Projects/learning_environments/results/GTNC_evaluate_reward_cartpole_2021-06-05-16_2'
LOG_DICT['5'] = '/home/ferreira/Projects/learning_environments/results/GTNC_evaluate_reward_cartpole_2021-06-06-13_5'
LOG_DICT['6'] = '/home/ferreira/Projects/learning_environments/results/GTNC_evaluate_reward_cartpole_2021-06-07-11_6'

MODEL_NUM = 10
MODEL_AGENTS = 10


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
    config['device'] = 'cpu'
    config['envs']['CartPole-v0']['solved_reward'] = 100000  # something big enough to prevent early out triggering

    env_factory = EnvFactory(config=config)
    reward_env = env_factory.generate_reward_env()
    reward_env.load_state_dict(save_dict['model'])
    real_env = env_factory.generate_real_env()

    return reward_env, real_env, config


def train_test_agents(mode, env, real_env, config):
    rewards = []
    episode_lengths = []

    # settings for comparability
    config['agents']['duelingddqn'] = {}
    config['agents']['duelingddqn']['test_episodes'] = 1
    config['agents']['duelingddqn']['train_episodes'] = 1000
    config['agents']['duelingddqn']['print_rate'] = 100

    config['agents']['duelingddqn']['lr'] = 0.00025
    config['agents']['duelingddqn']['eps_init'] = 1.0
    config['agents']['duelingddqn']['eps_min'] = 0.1
    config['agents']['duelingddqn']['eps_decay'] = 0.9  # original DDQN paper uses linear decay over 1M N's
    config['agents']['duelingddqn']['gamma'] = 0.99
    config['agents']['duelingddqn']['batch_size'] = 32
    config['agents']['duelingddqn']['same_action_num'] = 1
    config['agents']['duelingddqn']['activation_fn'] = "relu"
    config['agents']['duelingddqn']['tau'] = 0.01  # original DDQN paper has hard update every N steps
    config['agents']['duelingddqn']['hidden_size'] = 64
    config['agents']['duelingddqn']['hidden_layer'] = 1
    config['agents']['duelingddqn']['rb_size'] = 1000000
    config['agents']['duelingddqn']['init_episodes'] = 1
    config['agents']['duelingddqn']['feature_dim'] = 128

    config['agents']['duelingddqn']['early_out_num'] = 10
    config['agents']['duelingddqn']['early_out_virtual_diff'] = 0.02

    # optimized ICM HPs:
    config['agents']['icm'] = {}
    config['agents']['icm']['beta'] = 0.05
    config['agents']['icm']['eta'] = 0.03
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
            agent = select_agent(config=config, agent_name='duelingddqn_icm')
        else:
            agent = select_agent(config=config, agent_name='duelingddqn')
        reward, episode_length, _ = agent.train(env=env, test_env=real_env)
        rewards.append(reward)
        episode_lengths.append(episode_length)
    return rewards, episode_lengths


def save_list(mode, config, reward_list, episode_length_list):
    os.makedirs(SAVE_DIR, exist_ok=True)
    file_name = os.path.join(SAVE_DIR, 'best_transfer_algo' + str(mode) + '.pt')
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
