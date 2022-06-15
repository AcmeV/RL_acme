import json
import os

import torch
import  numpy as np
from torch import nn

from model.policy_gradient.net.ContinueNetwork import ContinueNetwork


class PolicyGradientContinue:

    def __init__(self, action_bound, n_features, n_hidden=20, learning_rate=0.01, reward_decay=0.95, device='cpu'):

        self.n_features = n_features
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.gamma = reward_decay
        self.action_bound = action_bound

        # episode_observations, episode_actions, episode_rewards
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self.device = device

        self._build_net()

    def _build_net(self):
        self.net = ContinueNetwork(self.n_features, self.n_hidden).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss(reduce=False).to(self.device)

    def choose_action(self, observation):
        observation = np.array(observation)[np.newaxis, :]
        normal_dist = self.normal_dist(observation)
        self.action = torch.clamp(normal_dist.sample(), self.action_bound[0], self.action_bound[1])  # sample action accroding to the distribution
        return self.action

    def normal_dist(self, s):
        s = torch.Tensor(s[np.newaxis, :]).to(self.device)
        mu, sigma = self.net(s)
        mu, sigma = (mu * 2).squeeze(), (sigma + 0.1).squeeze()
        normal_dist = torch.distributions.Normal(mu, sigma)  # get the normal distribution of average=mu and std=sigma
        return normal_dist

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        ep_loss = []
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        for i in range(len(self.ep_as)):
            a = self.ep_as[i]
            s = self.ep_obs[i]
            td = discounted_ep_rs_norm[i]

            normal_dist = self.normal_dist(s)
            log_prob = normal_dist.log_prob(a)  # log_prob get the probability of action a under the distribution of normal_dist
            exp_v = log_prob * float(td)  # advantage (TD_error) guided loss
            exp_v += 0.01 * normal_dist.entropy()  # Add cross entropy cost to encourage exploration
            loss = -exp_v  # max(v) = min(-v)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            ep_loss.append(loss.item())

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        return np.mean(ep_loss)

    def _discount_and_norm_rewards(self):
        # rewards decay in an episode
        discounted_ep_rs = np.zeros_like(self.ep_rs, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = float(running_add)

        if len(discounted_ep_rs) > 1:
            discounted_ep_rs -= np.mean(discounted_ep_rs)
            discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def save(self, episode, path):
        if not os.path.exists(path):
            os.makedirs(path)


        info_dict = {
            'episode': episode,
            'gamma': self.gamma
        }

        json.dump(info_dict, open(f'{path}/info.json', 'w'))
        torch.save(self.net.state_dict(), f'{path}/net.params')

    def load(self, path):
        self.net.load_state_dict(torch.load(f'{path}/net.params', map_location=self.device))
        info_json = json.load(open(f'{path}/info.json', 'r'))

        self.episode = info_json['episode']
        self.gamma = info_json['gamma']

        return int(info_json['episode'])