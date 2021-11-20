import os

import numpy as np

from utils import save_lists


def get_all_files(with_vary_hp, model_num, model_dir, custom_load_envs_and_config, env_name, device, filter_models_list=None):
    file_list = []
    for file_name in os.listdir(model_dir):
        if env_name not in file_name:
            continue

        _, _, config = custom_load_envs_and_config(file_name=file_name, model_dir=model_dir, device=device)
        if config['agents']['ddqn_vary']['vary_hp'] == with_vary_hp:
            file_list.append(file_name)

    # sort file list by random characters/digits -> make randomness deterministic
    file_list = sorted(file_list, key=lambda elem: elem[-9:])
    if len(file_list) < model_num and filter_models_list is None:
        raise ValueError("Not enough saved models")

    if filter_models_list is not None:
        filtered_file_list = [f for f in file_list if f in filter_models_list]
        print(f"model files filtered. Old number of models: {len(file_list)} new number of models: {len(filtered_file_list)}")
        print("used models: ", filtered_file_list)
        return filtered_file_list

    return file_list[:model_num]


def run_vary_hp(mode, experiment_name, model_num, agents_num, model_dir, custom_load_envs_and_config,
                custom_train_test_agents, env_name, pool=None, device="cuda", filter_models_list=None, correlation_exp=False):
    if mode == 0:
        train_on_venv = False
    elif mode == 1:
        train_on_venv = True
        with_vary_hp = False
    elif mode == 2:
        train_on_venv = True
        with_vary_hp = True

    env_reward_overview = {}
    reward_list = []
    train_steps_needed = []
    episode_length_needed = []

    if not train_on_venv:
        file_name = os.listdir(model_dir)[0]
        _, real_env, config = custom_load_envs_and_config(file_name=file_name, model_dir=model_dir, device=device)

        if pool is None:
            for i in range(model_num):
                print('train on {}-th environment'.format(i))
                reward_list_i, train_steps_needed_i, episode_length_needed_i = custom_train_test_agents(train_env=real_env,
                                                                                                        test_env=real_env,
                                                                                                        config=config,
                                                                                                        agents_num=agents_num
                                                                                                        )
                reward_list += reward_list_i
                train_steps_needed += train_steps_needed_i
                episode_length_needed += episode_length_needed_i
                
                if correlation_exp:
                    env_reward_overview[real_env.env.env_name + "_" + str(i)] = {}
                else:
                    env_reward_overview[real_env.env.env_name + "_" + str(i)] = np.hstack(reward_list_i)
        else:
            reward_list_tpl, train_steps_needed_tpl, episode_length_needed_tpl = zip(*pool.starmap(custom_train_test_agents,
                                                                                                   [(real_env, real_env, config,
                                                                                                     agents_num)
                                                                                                    for _ in range(model_num)])
                                                                                     )
            # starmap/map preservers order of calling
            for i in range(model_num):
                reward_list += reward_list_tpl[i]
                train_steps_needed += train_steps_needed_tpl[i]
                episode_length_needed += episode_length_needed_tpl[i]
                if correlation_exp:
                    env_reward_overview[real_env.env.env_name + "_" + str(i)] = {}
                else:
                    env_reward_overview[real_env.env.env_name + "_" + str(i)] = np.hstack(reward_list_tpl[i])
    else:
        file_list = get_all_files(with_vary_hp=with_vary_hp, model_num=model_num, model_dir=model_dir,
                                  custom_load_envs_and_config=custom_load_envs_and_config, env_name=env_name, device=device,
                                  filter_models_list=filter_models_list)

        if pool is None:
            for file_name in file_list:
                virtual_env, real_env, config = custom_load_envs_and_config(file_name=file_name, model_dir=model_dir, device=device)
                print('train agents on ' + str(file_name))

                reward_list_i, train_steps_needed_i, episode_length_needed_i = custom_train_test_agents(train_env=virtual_env,
                                                                                                        test_env=real_env,
                                                                                                        config=config,
                                                                                                        agents_num=agents_num
                                                                                                        )
                reward_list += reward_list_i
                train_steps_needed += train_steps_needed_i
                episode_length_needed += episode_length_needed_i
                if correlation_exp:
                    env_reward_overview[file_name] = {}
                else:
                    env_reward_overview[file_name] = np.hstack(reward_list_i)
        else:
            _, _, config = custom_load_envs_and_config(file_name=file_list[0], model_dir=model_dir, device=device)

            reward_list_tpl, train_steps_needed_tpl, episode_length_needed_tpl = zip(*pool.starmap(custom_train_test_agents,
                                                                                                   [(*custom_load_envs_and_config(file_name,
                                                                                                                                  model_dir,
                                                                                                                                  device),
                                                                                                     agents_num)
                                                                                                    for file_name in file_list]
                                                                                                   )
                                                                                     )
            # starmap/map preservers order of calling
            for i, file_name in enumerate(file_list):
                reward_list += reward_list_tpl[i]
                train_steps_needed += train_steps_needed_tpl[i]
                episode_length_needed += episode_length_needed_tpl[i]
                
                if correlation_exp:
                    env_reward_overview[file_name] = {}
                else:
                    env_reward_overview[file_name] = np.hstack(reward_list_tpl[i])

    save_lists(mode=mode,
               config=config,
               reward_list=reward_list,
               train_steps_needed=train_steps_needed,
               episode_length_needed=episode_length_needed,
               env_reward_overview=env_reward_overview,
               experiment_name=experiment_name
               )
