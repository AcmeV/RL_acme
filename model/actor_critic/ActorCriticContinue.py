import json
import os

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

    def normal_dist(self, s, device='cpu'):

        s = torch.Tensor(s[np.newaxis, :]).to(device)

        mu, sigma = self.net(s)

        mu, sigma = (mu * 2).squeeze(), (sigma + 0.1).squeeze()

        normal_dist = torch.distributions.Normal(mu, sigma)  # get the normal distribution of average=mu and std=sigma

        return normal_dist

    def _build_net(self):
        self.net = ContinueNetwork(self.n_features, self.n_hiddens)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def choose_action(self, s, device='cpu'):
        normal_dist = self.normal_dist(s, device)
        self.action = torch.clamp(normal_dist.sample(), self.action_bound[0],
                                  self.action_bound[1])  # sample action accroding to the distribution
        return self.action.cpu()

    def learn(self, s, a, td, device='cpu'):
        normal_dist = self.normal_dist(s, device)
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


    def learn(self, s, r, s_, device='cpu'):
        s, s_ = torch.Tensor(s[np.newaxis, :]).to(device), torch.Tensor(s_[np.newaxis, :]).to(device)
        v, v_ = self.net(s), self.net(s_)
        td_error = r + self.gamma * v_ - v
        loss = td_error ** 2

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        return td_error

class ActorCriticContinue():

    def __init__(self, action_bound, n_features, n_hiddens=30, learning_rate=0.01, reward_decay=0.9, device='cpu'):
        self.actor = Actor(n_features, action_bound, n_hiddens)
        self.critic = Critic(n_features, n_hiddens, learning_rate, reward_decay)
        self.device = device

        self.actor.net = self.actor.net.to(device)
        self.critic.net = self.critic.net.to(device)



    def learn(self, s, r, a, s_):
        td_error = self.critic.learn(s, r, s_, self.device)
        td_error = td_error.detach()
        return self.actor.learn(s, a, td_error, self.device)

    def choose_action(self, s):
        return self.actor.choose_action(s, self.device)

    def save(self, episode, path):
        if not os.path.exists(path):
            os.makedirs(path)

        info_dict = {
            'episode': episode
        }

        json.dump(info_dict, open(f'{path}/info.json', 'w'))
        torch.save(self.actor.net.state_dict(), f'{path}/actor.params')
        torch.save(self.critic.net.state_dict(), f'{path}/critic.params')

    def load(self, path):
        self.actor.net.load_state_dict(torch.load(f'{path}/actor.params', map_location=self.device))
        self.critic.net.load_state_dict(torch.load(f'{path}/critic.params', map_location=self.device))
        info_json = json.load(open(f'{path}/info.json', 'r'))

        self.episode = info_json['episode']

        return int(info_json['episode'])
