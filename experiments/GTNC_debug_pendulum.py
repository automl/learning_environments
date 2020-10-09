from agents.GTN import *
from agents.TD3 import TD3

if __name__ == "__main__":
    data = torch.load('/home/dingsda/master_thesis/learning_environments/agents/results/GTN_models/pendulum_LSVCZB.pt')
    env = data['model']
    config = data['config']
    config['agents']['td3']['print_rate'] = 1

    td3 = TD3(state_dim=env.get_state_dim(),
              action_dim=env.get_action_dim(),
              max_action=env.get_max_action(),
              config=config)

    td3.train(env=env, time_remaining=1200)
