import os
import sys

import hpbandster.core.result as hpres
import torch

from agents.agent_utils import select_agent
from envs.env_factory import EnvFactory

# SAVE_DIR = '/home/nierhoff/master_thesis/learning_environments/results/cliff_compare_reward_envs'
SAVE_DIR = '/home/ferreira/Projects/learning_environments/results/cliff_compare_reward_envs'

LOG_DICT = {}
LOG_DICT['1'] = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_cliff_2021-02-09-21_1'
LOG_DICT['2'] = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_cliff_2021-02-09-21_2'
LOG_DICT['5'] = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_cliff_2021-02-09-21_5'
LOG_DICT['6'] = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_cliff_2021-02-09-21_6'

# LOG_DICT['1'] = '/home/ferreira/Projects/learning_environments/results/thomas_results/GTNC_evaluate_cliff_2021-02-09-21_1'
# LOG_DICT['2'] = '/home/ferreira/Projects/learning_environments/results/thomas_results/GTNC_evaluate_cliff_2021-02-09-21_2'
# LOG_DICT['5'] = '/home/ferreira/Projects/learning_environments/results/thomas_results/GTNC_evaluate_cliff_2021-02-09-21_5'
# LOG_DICT['6'] = '/home/ferreira/Projects/learning_environments/results/thomas_results/GTNC_evaluate_cliff_2021-02-09-21_6'


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
    # best_models.sort(key=lambda x: x[0], reverse=True)
    best_models = best_models[:MODEL_NUM]

    return best_models


def load_envs_and_config(model_file):
    save_dict = torch.load(model_file)

    config = save_dict['config']
    # config['device'] = 'cuda'
    config['envs']['Cliff']['solved_reward'] = 100000  # something big enough to prevent early out triggering

    env_factory = EnvFactory(config=config)
    reward_env = env_factory.generate_reward_env()
    reward_env.load_state_dict(save_dict['model'])
    real_env = env_factory.generate_real_env()

    return reward_env, real_env, config


def train_test_agents(mode, env, real_env, config):
    rewards = []
    episode_lengths = []

    # settings for comparability
    config['agents']['sarsa'] = {}
    config['agents']['sarsa']['test_episodes'] = 1
    config['agents']['sarsa']['train_episodes'] = 500
    config['agents']['sarsa']['print_rate'] = 100

    config['agents']['sarsa']['alpha'] = 1.0
    config['agents']['sarsa']['eps_decay'] = 0.0
    config['agents']['sarsa']['eps_init'] = 0.01
    config['agents']['sarsa']['eps_min'] = 0.01
    config['agents']['sarsa']['gamma'] = 0.8
    config['agents']['sarsa']['same_action_num'] = 1
    config['agents']['sarsa']['rb_size'] = 1  # custom to reward env and gridworld
    config['agents']['sarsa']['init_episodes'] = 0
    config['agents']['sarsa']['batch_size'] = 1

    config['agents']['sarsa']['early_out_num'] = 10
    config['agents']['sarsa']['early_out_virtual_diff'] = 0.02

    # for count-based q-learning
    config['agents']['sarsa']['beta'] = 0.1

    # for count-based q-learning (tuned)
    # config['agents']['sarsa']['beta'] = 0.005  # 0.01 also works fine

    for i in range(MODEL_AGENTS):
        if mode == '-1':
            agent = select_agent(config=config, agent_name='sarsa_cb')
        else:
            agent = select_agent(config=config, agent_name='sarsa')
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
        # if not os.path.isdir(best_models[0][1]):
        #     model_file = best_models[0][1].replace('/home/dingsda/master_thesis/learning_environments/results',
        #                                            '/home/ferreira/Projects/learning_environments/results/thomas_results')
        # else:
        #     model_file = best_models[0][1]

        reward_env, real_env, config = load_envs_and_config(best_models[0][1])
        rewards, episode_lengths = train_test_agents(mode=mode, env=real_env, real_env=real_env, config=config)
        reward_list += rewards
        episode_length_list += episode_lengths

    save_list(mode, config, reward_list, episode_length_list)


if __name__ == "__main__":

    for arg in sys.argv[1:]:
        print(arg)
    mode = str(sys.argv[1])

    if mode == '0' or mode == '-1':
        eval_base(mode=mode, log_dir=LOG_DICT['2'])
    else:
        eval_models(mode=mode, log_dir=LOG_DICT[mode])
