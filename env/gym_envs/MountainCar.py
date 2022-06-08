import gym
import gym.envs.classic_control.mountain_car
import numpy as np


class MountainCar():

    def __init__(self):

        self.env = gym.make("MountainCar-v0").unwrapped

        self.n_actions = 3

        self.has_terminal_tag = True

        self.is_render = False

        self.n_features = self.env.observation_space.shape[0]

    def reset(self):
        return self.env.reset(seed=21)

    def render(self):
        if self.is_render:
            self.env.render()
        else:
            pass

    def step(self, action):

        observation_, reward, done, info = self.env.step(action)

        if done:
            reward = 100

        return observation_, reward, done, info

    def close(self):
        self.env.close()
