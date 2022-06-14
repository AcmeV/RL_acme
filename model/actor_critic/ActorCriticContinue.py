import numpy as np
import torch

from model.actor_critic.net.ContinueNetwork import ContinueNetwork
from model.actor_critic.net.DiscreteNetwork import DiscreteNetwork


class Actor(object):
    def __init__(self, n_features, action_bound, n_hiddens=30, learning_rate=0.001):
        self.n_features = n_features
        self.action_bound = action_bound
        self.n_hiddens = n_hiddens
        self.lr = learning_rate

        self._build_net()

    def normal_dist(self, s):
        s = torch.Tensor(s[np.newaxis, :])
        mu, sigma = self.net(s)
        mu, sigma = (mu * 2).squeeze(), (sigma + 0.1).squeeze()
        normal_dist = torch.distributions.Normal(mu, sigma)  # get the normal distribution of average=mu and std=sigma
        return normal_dist

    def _build_net(self):
        self.net = ContinueNetwork(self.n_features, self.n_hiddens)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def choose_action(self, s):
        normal_dist = self.normal_dist(s)
        self.action = torch.clamp(normal_dist.sample(), self.action_bound[0],
                                  self.action_bound[1])  # sample action accroding to the distribution
        return self.action

    def learn(self, s, a, td):
        normal_dist = self.normal_dist(s)
        # log_prob get the probability of action a under the distribution of normal_dist
        log_prob = normal_dist.log_prob(a)
        exp_v = log_prob * td.float()  # advantage (TD_error) guided loss
        exp_v += 0.01 * normal_dist.entropy()  # Add cross entropy cost to encourage exploration
        loss = -exp_v  # max(v) = min(-v)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


class Critic(object):

    def __init__(self, n_features, n_hiddens=30, learning_rate=0.01, reward_decay=0.9):

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

class ActorCriticContinue():

    def __init__(self, action_bound, n_features, n_hiddens=30, learning_rate=0.01, reward_decay=0.9):
        self.actor = Actor(n_features, action_bound, n_hiddens)
        self.critic = Critic(n_features, n_hiddens, learning_rate, reward_decay)
        self.gamma = reward_decay
        self.actor_net = self.actor.net
        self.critic_net = self.critic.net
        self.actor_optim = self.actor.optimizer
        self.critic_optim = self.critic.optimizer


    def learn(self, s, r, a, s_):
        td_error = self.critic.learn(s, r, s_)
        td_error = td_error.detach()
        return self.actor.learn(s, a, td_error)

    def choose_action(self, s):
        return self.actor.choose_action(s)
