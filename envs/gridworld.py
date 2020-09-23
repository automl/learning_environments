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


G_LEFT = 0
G_UP = 1
G_RIGHT = 2
G_DOWN = 3


class GridworldEnv(gym.Env):
    """
    Bandit environment base to allow agents to interact with the class n-armed bandit
    in different variations
    p_dist:
        A list of probabilities of the likelihood that a particular bandit will pay out
    r_dist:
        A list of either rewards (if number) or means and standard deviations (if list)
        of the payout that bandit has
    """
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
            ac = 'LEFT'
            x_n, y_n = x, y-1
        elif action == G_UP:
            ac = 'UP'
            x_n, y_n = x-1, y
        elif action == G_RIGHT:
            ac = 'RIGHT'
            x_n, y_n = x, y+1
        elif action == G_DOWN:
            ac = 'DOWN'
            x_n, y_n = x+1, y
        else:
            raise ValueError('Unknown action: ' + str(action))

        # movement limited by map boundaries
        x_n = min(max(x_n, 0), m-1)
        y_n = min(max(y_n, 0), n-1)

        # movement limited by wall boundaries
        if self.grid[x_n][y_n] == '#':
            x_n, y_n = x, y

        #print('{} {} -> {} -> {} {}'.format(x, y, ac, x_n, y_n))

        return x_n, y_n

    # calculate reward flag
    def _calc_reward(self):
        x, y = self.state
        if self.grid[x][y] == 'G':
            #print('reward 1')
            return 1
        elif self.grid[x][y] == 'O':
            #print('reward -1')
            return -1
        else:
            return 0

    # calculate done flag
    def _calc_done(self):
        x, y = self.state
        if self.grid[x][y] in ('O', 'G'):
            #print('done')
            return True
        else:
            return False

    # convert from observation (int) to internal state representation (x,y)
    def _obs_to_state(self, obs):
        m = len(self.grid)
        n = len(self.grid[0])

        x = obs // m
        y = obs % n
        return x,y

    # convert from internal state representation (x,y) to observation (int)
    def _state_to_obs(self, state):
        m = len(self.grid)
        x,y = state
        obs = x*m + y
        return obs


class EmptyRoom(GridworldEnv):
    def __init__(self):
        grid = [['S', ' ', ' ', ' ', ' '],
                [' ', ' ', ' ', ' ', ' '],
                [' ', ' ', ' ', ' ', ' '],
                [' ', ' ', ' ', ' ', ' '],
                [' ', ' ', ' ', ' ', 'G']]

        GridworldEnv.__init__(self, grid=grid)


class WallRoom(GridworldEnv):
    def __init__(self):
        grid = [['S', ' ', '#', ' ', ' '],
                [' ', ' ', '#', ' ', ' '],
                [' ', ' ', ' ', ' ', ' '],
                [' ', ' ', '#', ' ', ' '],
                [' ', ' ', '#', ' ', 'G']]

        GridworldEnv.__init__(self, grid=grid)


class HoleRoom(GridworldEnv):
    def __init__(self):
        grid = [['S', ' ', 'O', ' ', ' '],
                [' ', ' ', 'O', ' ', ' '],
                [' ', ' ', ' ', ' ', ' '],
                [' ', ' ', 'O', ' ', ' '],
                [' ', ' ', 'O', ' ', 'G']]

        GridworldEnv.__init__(self, grid=grid)

class Cliff(GridworldEnv):
    def __init__(self):
        grid = [[' ', ' ', ' ', ' ', ' '],
                [' ', ' ', ' ', ' ', ' '],
                ['S', 'O', 'O', 'O', 'G']]

        GridworldEnv.__init__(self, grid=grid)

