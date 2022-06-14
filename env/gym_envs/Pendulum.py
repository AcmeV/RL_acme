import gym
import numpy as np


class Pendulum():

    def __init__(self):

        self.env = gym.make("Pendulum-v1").unwrapped

        self.n_actions = 11

        self.has_terminal_tag = False

        self.is_render = False

        self.discrete = True

        self.n_features = self.env.observation_space.shape[0]

    def reset(self):
        return self.env.reset(seed=1)

    def render(self):
        if self.is_render:
            self.env.render()
        else:
            pass

    def step(self, action):

        f_action = (action - (self.n_actions - 1) / 2) / ((self.n_actions - 1) / 4)

        observation_, reward, done, info = self.env.step(np.array([f_action]))

        # normalize to a range of (-1,0). r = 0 when get upright
        reward /= 10

        return observation_, reward, done, info

    def close(self):
        self.env.close()
