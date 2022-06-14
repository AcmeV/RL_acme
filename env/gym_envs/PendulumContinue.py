import gym
import numpy as np


class PendulumContinue():

    def __init__(self):

        self.env = gym.make("Pendulum-v1").unwrapped

        self.action_bound = [-2.0, 2.0]

        self.has_terminal_tag = False

        self.is_render = False

        self.discrete = False

        self.n_features = self.env.observation_space.shape[0]

    def reset(self):
        return self.env.reset(seed=1)

    def render(self):
        if self.is_render:
            self.env.render()
        else:
            pass

    def step(self, action):

        observation_, reward, done, info = self.env.step(np.array([action]))

        # normalize to a range of (-1,0). r = 0 when get upright
        reward /= 10

        return observation_, reward, done, info

    def close(self):
        self.env.close()
