import numpy as np
import torch

from model.actor_critic.net.DiscreteNetwork import DiscreteNetwork


class Actor(object):
    def __init__(self, n_features, n_actions, n_hiddens=20, learning_rate=0.001):
        self.n_features = n_features
        self.n_actions = n_actions
        self.n_hiddens = n_hiddens
        self.lr = learning_rate

        self._build_net()


    def _build_net(self):
        self.net = DiscreteNetwork(self.n_features, self.n_hiddens, self.n_actions, activate=True)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)


    def choose_action(self, s):
        s = torch.Tensor(s[np.newaxis, :])
        probs = self.net(s)
        return np.random.choice(np.arange(probs.shape[1]), p=probs.data.numpy().ravel())


    def learn(self, s, a, td):
        s = torch.Tensor(s[np.newaxis, :])
        acts_prob = self.net(s)
        log_prob = torch.log(acts_prob[0, a])
        exp_v = torch.mean(log_prob * td)

        loss = -exp_v
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        return loss.item()


class Critic(object):

    def __init__(self, n_features, n_hiddens, learning_rate=0.01, reward_decay=0.9):

        self.n_features = n_features
        self.gamma = reward_decay
        self.n_hiddens = n_hiddens
        self.lr = learning_rate
        self._build_net()


    def _build_net(self):
        self.net = DiscreteNetwork(self.n_features, self.n_hiddens, 1)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)


    def learn(self, s, r, s_):
        s, s_ = torch.Tensor(s[np.newaxis, :]), torch.Tensor(s_[np.newaxis, :])
        v, v_ = self.net(s), self.net(s_)
        td_error = r + self.gamma * v_ - v
        loss = td_error ** 2

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        return td_error

class ActorCriticDiscrete():

    def __init__(self, n_actions, n_features, n_hiddens=20, learning_rate=0.01, reward_decay=0.9):
        self.actor = Actor(n_features, n_actions, n_hiddens)
        self.critic = Critic(n_features, n_hiddens, learning_rate, reward_decay)

    def learn(self, s, r, a, s_):
        td_error = self.critic.learn(s, r, s_)
        td_error = td_error.detach()
        return self.actor.learn(s, a, td_error)

    def choose_action(self, s):
        return self.actor.choose_action(s)
