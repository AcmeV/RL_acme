import json
import os

import torch
import  numpy as np
from torch import nn

from model.policy_gradient.net.DiscreteNetwork import DiscreteNetwork


class PolicyGradientDiscrete:

    def __init__(self, n_actions, n_features, n_hidden=20, learning_rate=0.01, reward_decay=0.95, device='cpu'):

        self.n_actions = n_actions
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.gamma = reward_decay

        # episode_observations, episode_actions, episode_rewards
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self.device = device

        self._build_net()

    def _build_net(self):
        self.net = DiscreteNetwork(self.n_features, self.n_hidden, self.n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss(reduce=False).to(self.device)

    def choose_action(self, observation):

        observation = torch.Tensor(np.array(observation)[np.newaxis, :]).to(self.device)
        prob = self.net(observation)
        action = np.random.choice(range(prob.shape[1]), p=prob.cpu().data.numpy().ravel())
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        obs = torch.Tensor(np.vstack(self.ep_obs)).to(self.device)
        acts = torch.Tensor(np.array(self.ep_as)).to(self.device)
        vt = torch.Tensor(discounted_ep_rs_norm).to(self.device)

        all_act = self.net(obs)

        # cross_entropy combines nn.LogSoftmax() and nn.NLLLoss() in one single class
        neg_log_prob = self.criterion(all_act, acts.long())
        loss = torch.mean(neg_log_prob * vt)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        return loss.item()

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
