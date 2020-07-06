import yaml
from sklearn.svm import SVR
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from joblib import parallel_backend
from envs.env_factory import EnvFactory
from agents.match_env import *

TRAIN_SAMPLES = 100000
TEST_SAMPLES = 1000

def train(real_env):
    inputs_list = []
    outputs_list = []

    real_env.reset()

    with parallel_backend('threading', n_jobs=8):
        for k in range(TRAIN_SAMPLES):
            # run random state/actions transitions on the real env
            state = real_env.env.observation_space.sample()*1.1 # the 1.1 is important!
            action = real_env.env.action_space.sample()*1.1     # the 1.1 is important!
            next_state, reward, done = real_env.step(
                action=torch.tensor(action, device=device, dtype=torch.float32),
                state=torch.tensor(state, device=device, dtype=torch.float32))
            inputs_list.append(np.concatenate([state, action]))
            outputs_list.append(np.concatenate([next_state.cpu().data.numpy(),
                                                reward.unsqueeze(0).cpu().data.numpy(),
                                                done.unsqueeze(0).cpu().data.numpy()]))

            if k % 1000 == 0:
                print(k)

        input = np.asarray(inputs_list)
        output = np.asarray(outputs_list)

        # SVR
        # svr = SVR()
        # model = MultiOutputRegressor(svr)

        # random forest
        # model = RandomForestRegressor()

        # GBR
        # gbr = GradientBoostingRegressor()
        # model = MultiOutputRegressor(gbr)

        model.fit(input, output)

    return model


def validate(real_env, model):
    inputs_list = []
    outputs_list = []

    for k in range(TEST_SAMPLES):
        # run random state/actions transitions on the real env
        state = real_env.env.observation_space.sample()*1.1 # the 1.1 is important!
        action = real_env.env.action_space.sample()*1.1     # the 1.1 is important!
        next_state, reward, done = real_env.step(
            action=torch.tensor(action, device=device, dtype=torch.float32),
            state=torch.tensor(state, device=device, dtype=torch.float32))
        inputs_list.append(np.concatenate([state, action]))
        outputs_list.append(np.concatenate([next_state.cpu().data.numpy(),
                                            reward.unsqueeze(0).cpu().data.numpy(),
                                            done.unsqueeze(0).cpu().data.numpy()]))

    input = np.asarray(inputs_list)
    output_real = np.asarray(outputs_list)
    output_pred = model.predict(input)

    abs_diff = np.sum(abs(output_pred-output_real), axis=0) / TEST_SAMPLES

    print(abs_diff)


if __name__ == "__main__":
    with open("../default_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    config['env_name'] = 'Test'
    config['env_name'] = 'Pendulum-v0'

    # generate environment
    env_fac = EnvFactory(config)
    real_env = env_fac.generate_default_real_env()

    model = train(real_env)
    validate(real_env, model)

