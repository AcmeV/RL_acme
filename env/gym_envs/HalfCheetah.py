import gym

class HalfCheetah():

    def __init__(self):

        self.env = gym.make("HalfCheetah-v2").unwrapped

        self.action_bound = [-1.0, 1.0]

        self.has_terminal_tag = False

        self.is_render = False

        self.discrete = False
        self.n_features = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

    def reset(self):
        return self.env.reset(seed=1)

    def render(self):
        if self.is_render:
            self.env.render()
        else:
            pass

    def step(self, action):

        observation_, reward, done, info = self.env.step(action)

        return observation_, reward, done, info

    def close(self):
        self.env.close()
