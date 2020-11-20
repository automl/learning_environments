import numpy as np
import gym
import random
from gym import spaces
from gym.utils import seeding


# 'S': start
# 'G': goal
# ' ': empty cell
# '#': wall
# 'O': hole

G_RIGHT = 0
G_LEFT = 1
G_DOWN = 2
G_UP = 3

# G_LEFT = 0
# G_UP = 1
# G_RIGHT = 2
# G_DOWN = 3

class GridworldEnv(gym.Env):
    def __init__(self, grid):
        m = len(grid)
        n = len(grid[0])
        self.grid = grid
        self.state = None
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(m*n)
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)

        self.state = self._calc_next_state(action)
        reward = self._calc_reward()
        done = self._calc_done()

        obs = self._state_to_obs(self.state)
        return obs, reward, done, {}

    def reset(self):
        m = len(self.grid)
        n = len(self.grid[0])

        # find start state
        for x in range(m):
            for y in range(n):
                if self.grid[x][y] == 'S':
                    self.state = (x,y)
                    return self._state_to_obs(self.state)

        raise ValueError("No start state found")

    def render(self, mode='human', close=False):
        pass

    def _calc_next_state(self, action):
        m = len(self.grid)
        n = len(self.grid[0])
        x, y = self.state

        # hole -> stuck
        if self.grid[x][y] == 'O':
            return x, y

        # calculate movement direction
        if action == G_LEFT:
            x_n, y_n = x, y-1
        elif action == G_UP:
            x_n, y_n = x-1, y
        elif action == G_RIGHT:
            x_n, y_n = x, y+1
        elif action == G_DOWN:
            x_n, y_n = x+1, y
        else:
            raise ValueError('Unknown action: ' + str(action))

        # movement limited by map boundaries
        x_n = min(max(x_n, 0), m-1)
        y_n = min(max(y_n, 0), n-1)

        # movement limited by wall boundaries
        if self.grid[x_n][y_n] == '#':
            x_n, y_n = x, y

        return x_n, y_n

    # calculate reward flag
    def _calc_reward(self):
        x, y = self.state
        if self.grid[x][y] == 'G':
            return self.g_reward
        elif self.grid[x][y] == 'O':
            return self.o_reward
        else:
            return self.step_cost

    # calculate done flag
    def _calc_done(self):
        x, y = self.state
        if self.grid[x][y] in ('O', 'G'):
            return True
        else:
            return False

    # convert from observation (int) to internal state representation (x,y)
    def _obs_to_state(self, obs):
        n = len(self.grid[0])
        x = obs // n
        y = obs % n
        return x,y

    # convert from internal state representation (x,y) to observation (int)
    def _state_to_obs(self, state):
        n = len(self.grid[0])
        x,y = state
        obs = x*n + y
        return obs


class EmptyRoom22(GridworldEnv):
    def __init__(self):
        self.step_cost = -0.01
        self.g_reward = 1
        self.o_reward = -1
        grid = [['S', ' '],
                [' ', 'G']]
        GridworldEnv.__init__(self, grid=grid)


class EmptyRoom23(GridworldEnv):
    def __init__(self):
        self.step_cost = -0.01
        self.g_reward = 1
        self.o_reward = -1
        grid = [['S', ' ', ' '],
                [' ', ' ', 'G']]
        GridworldEnv.__init__(self, grid=grid)

class EmptyRoom33(GridworldEnv):
    def __init__(self):
        self.step_cost = -0.01
        self.g_reward = 1
        self.o_reward = -1
        grid = [['S', ' ', ' '],
                [' ', ' ', ' '],
                [' ', ' ', 'G']]
        GridworldEnv.__init__(self, grid=grid)


class EmptyRoom(GridworldEnv):
    def __init__(self):
        self.step_cost = -0.01
        self.g_reward = 1
        self.o_reward = -1
        grid = [['S', ' ', ' ', ' ', ' '],
                [' ', ' ', ' ', ' ', ' '],
                [' ', ' ', ' ', ' ', ' '],
                [' ', ' ', ' ', ' ', ' '],
                [' ', ' ', ' ', ' ', 'G']]

        GridworldEnv.__init__(self, grid=grid)


class WallRoom(GridworldEnv):
    def __init__(self):
        self.step_cost = -0.01
        self.g_reward = 1
        self.o_reward = -1
        grid = [['S', ' ', '#', ' ', ' ', ' '],
                [' ', ' ', '#', ' ', ' ', ' '],
                [' ', ' ', ' ', ' ', ' ', ' '],
                [' ', ' ', '#', ' ', ' ', ' '],
                [' ', ' ', '#', ' ', ' ', 'G']]

        GridworldEnv.__init__(self, grid=grid)


class HoleRoom(GridworldEnv):
    def __init__(self):
        self.step_cost = -0.01
        self.g_reward = 1
        self.o_reward = -1
        grid = [['S', ' ', 'O', ' ', 'O', ' ', ' '],
                [' ', ' ', 'O', ' ', 'O', ' ', ' '],
                [' ', ' ', ' ', ' ', ' ', ' ', ' '],
                [' ', ' ', 'O', ' ', 'O', ' ', ' '],
                [' ', ' ', 'O', ' ', 'O', ' ', 'G']]

        GridworldEnv.__init__(self, grid=grid)

class Cliff(GridworldEnv):
    def __init__(self):
        self.step_cost = -0.01
        self.g_reward = 0
        self.o_reward = -1
        grid = [[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                ['S', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'G']]

        GridworldEnv.__init__(self, grid=grid)

if __name__ == "__main__":
    r = Cliff()

    for i in range(25):
        x,y=r._obs_to_state(i)
        print('{} {} {} {}'.format(i,r._state_to_obs((x,y)), x, y))
        #print(r.grid[x][y])
