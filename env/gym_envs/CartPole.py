import gym
import gym.envs.classic_control.cartpole


class CartPole():

    def __init__(self):

        self.env = gym.make("CartPole-v0").unwrapped

        self.n_actions = 2

        self.has_terminal_tag = True

        self.is_render = False

        self.n_features = self.env.observation_space.shape[0]

    def reset(self):
        return self.env.reset(seed=1)

    def render(self):
        if self.is_render:
            self.env.render()
        else:
            pass

    def step(self, action):

        observation_, reward, done, info = self.env.step(action)

        x, xdot, theta, thata_dot = observation_
        r1 = (self.env.x_threshold - abs(x)) / self.env.x_threshold - 0.8
        r2 = (self.env.theta_threshold_radians - abs(theta)) / self.env.theta_threshold_radians - 0.5

        reward = r1 + r2
        return observation_, reward, done, info
