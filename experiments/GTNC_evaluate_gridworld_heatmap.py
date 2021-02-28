import os
import sys

import torch
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis

from agents.agent_utils import select_agent
from envs.env_factory import EnvFactory

SAVE_DIR = '/home/dingsda/master_thesis/learning_environments/results/cliff_compare_reward_envs'

LOG_DICT = {}
LOG_DICT['1'] = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_cliff_2021-02-09-21_1'
LOG_DICT['2'] = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_cliff_2021-02-09-21_2'
LOG_DICT['5'] = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_cliff_2021-02-09-21_5'
LOG_DICT['6'] = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_cliff_2021-02-09-21_6'

MODEL_NUM = 10
MODEL_AGENTS = 10
MODE = '6'       # '1', '2', '5', '6'
BREAK='solved'   # 'solved' or 'end'

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
    if BREAK == 'solved':
        config['envs']['Cliff']['solved_reward'] = -20  # something big enough to prevent early out triggering
    else:
        config['envs']['Cliff']['solved_reward'] = 100000  # something big enough to prevent early out triggering

    env_factory = EnvFactory(config=config)
    reward_env = env_factory.generate_reward_env()
    reward_env.load_state_dict(save_dict['model'])
    real_env = env_factory.generate_real_env()

    return reward_env, real_env, config


def train_test_agents(env, real_env, config):
    states = []

    # settings for comparability
    config['agents']['sarsa'] = {}
    config['agents']['sarsa']['test_episodes'] = 1
    config['agents']['sarsa']['train_episodes'] = 200
    config['agents']['sarsa']['print_rate'] = 100
    config['agents']['sarsa']['init_episodes'] = config['agents']['ql']['init_episodes']
    config['agents']['sarsa']['batch_size'] = config['agents']['ql']['batch_size']
    config['agents']['sarsa']['alpha'] = config['agents']['ql']['alpha']
    config['agents']['sarsa']['gamma'] = config['agents']['ql']['gamma']
    config['agents']['sarsa']['eps_init'] = config['agents']['ql']['eps_init']
    config['agents']['sarsa']['eps_min'] = config['agents']['ql']['eps_min']
    config['agents']['sarsa']['eps_decay'] = config['agents']['ql']['eps_decay']
    config['agents']['sarsa']['rb_size'] = config['agents']['ql']['rb_size']
    config['agents']['sarsa']['same_action_num'] = config['agents']['ql']['same_action_num']
    config['agents']['sarsa']['early_out_num'] = config['agents']['ql']['early_out_num']
    config['agents']['sarsa']['early_out_virtual_diff'] = config['agents']['ql']['early_out_virtual_diff']

    for i in range(MODEL_AGENTS):
        agent = select_agent(config=config, agent_name='sarsa')
        reward, _, _ = agent.train(env=env, test_env=real_env)
        _, _, replay_buffer = agent.test(env=real_env)
        state, _, next_state, _, _ = replay_buffer.get_all()
        state = state.tolist()
        next_state = next_state.tolist()
        # skip if we could not solve env
        if len(reward) == config['agents']['sarsa']['train_episodes'] and BREAK == 'solved':
            continue
        state = [int(elem[0]) for elem in state]
        state.append(int(next_state[-1][0]))
        states.append(state)

    return states


def save_list(config, state_list):
    os.makedirs(SAVE_DIR, exist_ok=True)
    file_name = os.path.join(SAVE_DIR, 'heatmap_' + str(BREAK) + '_' + str(MODE) + '.pt')
    save_dict = {}
    save_dict['config'] = config
    save_dict['model_num'] = MODEL_NUM
    save_dict['model_agents'] = MODEL_AGENTS
    save_dict['mode'] = MODE
    save_dict['break'] = BREAK
    save_dict['state_list'] = state_list
    torch.save(save_dict, file_name)


def eval_models(log_dir):
    best_models = get_best_models_from_log(log_dir)

    state_list = []

    for best_reward, model_file in best_models:
        print('best reward: ' + str(best_reward))
        print('model file: ' + str(model_file))
        reward_env, real_env, config = load_envs_and_config(model_file)
        states = train_test_agents(env=reward_env, real_env=real_env, config=config)
        state_list += states

    print(state_list)

    save_list(config, state_list)


if __name__ == "__main__":
    eval_models(log_dir=LOG_DICT[MODE])

