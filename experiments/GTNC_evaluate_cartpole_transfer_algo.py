import os
import sys

import torch
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis

from agents.agent_utils import select_agent
from envs.env_factory import EnvFactory

#SAVE_DIR = '/home/nierhoff/master_thesis/learning_environments/results/cartpole_compare_reward_envs'
SAVE_DIR = '/home/ferreira/Projects/learning_environments/results/cartpole_compare_reward_envs'

LOG_DICT = {}
LOG_DICT['1'] = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_cartpole_2021-01-28-18_1'
LOG_DICT['2'] = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_cartpole_2021-01-28-18_2'
LOG_DICT['5'] = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_cartpole_2021-01-28-18_5'
LOG_DICT['6'] = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_cartpole_2021-01-28-18_6'

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

    best_models.sort(key=lambda x: x[0])
    #best_models.sort(key=lambda x: x[0], reverse=True)
    best_models = best_models[:MODEL_NUM]

    return best_models


def load_envs_and_config(model_file):
    save_dict = torch.load(model_file)

    config = save_dict['config']
    config['device'] = 'cuda'
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
    config['agents']['duelingddqn']['init_episodes'] = config['agents']['ddqn']['init_episodes']
    config['agents']['duelingddqn']['batch_size'] = config['agents']['ddqn']['batch_size']
    config['agents']['duelingddqn']['gamma'] = config['agents']['ddqn']['gamma']
    config['agents']['duelingddqn']['lr'] = config['agents']['ddqn']['lr']
    config['agents']['duelingddqn']['tau'] = config['agents']['ddqn']['tau']
    config['agents']['duelingddqn']['eps_init'] = config['agents']['ddqn']['eps_init']
    config['agents']['duelingddqn']['eps_min'] = config['agents']['ddqn']['eps_min']
    config['agents']['duelingddqn']['eps_decay'] = config['agents']['ddqn']['eps_decay']
    config['agents']['duelingddqn']['rb_size'] = config['agents']['ddqn']['rb_size']
    config['agents']['duelingddqn']['same_action_num'] = config['agents']['ddqn']['same_action_num']
    config['agents']['duelingddqn']['activation_fn'] = config['agents']['ddqn']['activation_fn']
    config['agents']['duelingddqn']['hidden_size'] = config['agents']['ddqn']['hidden_size']
    config['agents']['duelingddqn']['hidden_layer'] = config['agents']['ddqn']['hidden_layer']
    config['agents']['duelingddqn']['feature_dim'] = 128
    config['agents']['duelingddqn']['early_out_num'] = config['agents']['ddqn']['early_out_num']
    config['agents']['duelingddqn']['early_out_virtual_diff'] = config['agents']['ddqn']['early_out_virtual_diff']

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

