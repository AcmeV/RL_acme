import json
import os
import random

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from model.ddpg.net import ActorNetwork, CriticNetwork
from model.ddpg.utils import OrnsteinUhlenbeckProcess
from model.ddpg.utils.convert import to_tensor


class DDPG():
    def __init__(self, action_bound, n_actions, n_features, n_hiddens=256,
                 learning_rate=0.01, reward_decay=0.99, tau=0.01,
                 memory_size=2000, batch_size=32, epsilon_decrement=1/20000,
                 device='cpu'):

        self.action_bound = action_bound
        self.n_features = n_features
        self.n_hiddens = n_hiddens
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.tau = tau

        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_decrement = epsilon_decrement
        self.epsilon = 1. if epsilon_decrement is not None else 0.

        self.device = device

        self.actor = ActorNetwork(self.n_features, self.n_hiddens, self.n_actions)
        self.actor_target = ActorNetwork(self.n_features, self.n_hiddens, self.n_actions)
        self.critic = CriticNetwork(self.n_features + self.n_actions, self.n_hiddens, self.n_actions)
        self.critic_target = CriticNetwork(self.n_features + self.n_actions, self.n_hiddens, self.n_actions)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.buffer = []
        self.memory = np.zeros((self.memory_size, n_features * 2 + 1 + self.n_actions))

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.random_process = OrnsteinUhlenbeckProcess(size=n_actions, theta=0.15, mu=0,
                                                       sigma=0.2)


    def store_transition(self, *transition):

        if len(self.buffer) == self.memory_size:
            self.buffer.pop(0)
        self.buffer.append(transition)


    def choose_action(self, state):

        s0 = torch.tensor(np.array(state), dtype=torch.float).unsqueeze(0)

        action = self.actor(s0).detach().cpu().data.numpy().squeeze(0)

        noise = np.random.uniform(self.action_bound[0], self.action_bound[1], self.n_actions)

        action += noise


        if self.epsilon > 0:
            self.epsilon -= self.epsilon_decrement
        action = np.clip(action, self.action_bound[0], self.action_bound[1])

        return action

    def learn(self):

        samples = random.sample(self.buffer, self.batch_size)

        s0, a0, r1, s1 = zip(*samples)

        s0 = torch.tensor(s0, dtype=torch.float).to(self.device)
        a0 = torch.tensor(a0, dtype=torch.float).to(self.device).view(self.batch_size, -1)
        r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size, -1).to(self.device)
        s1 = torch.tensor(s1, dtype=torch.float).to(self.device)

        def critic_learn():
            a1 = self.actor_target(s1).detach()
            y_true = r1 + self.gamma * self.critic_target(s1, a1).detach()

            y_pred = self.critic(s0, a0)

            loss_fn = nn.MSELoss()
            loss = loss_fn(y_pred, y_true)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()

            return loss.item()

        def actor_learn():
            loss = -torch.mean(self.critic(s0, self.actor(s0)))
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()
            return loss.item()

        loss = critic_learn()
        actor_learn()
        self.soft_update(self.critic_target, self.critic, )
        self.soft_update(self.actor_target, self.actor)
        return loss

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def save(self, episode, path):
        if not os.path.exists(path):
            os.makedirs(path)


        info_dict = {
            'episode': episode,
            'batch_size': self.batch_size,
            'epsilon_decrement': self.epsilon_decrement,
            'epsilon': self.epsilon,
            'gamma': self.gamma
        }

        json.dump(info_dict, open(f'{path}/info.json', 'w'))
        torch.save(self.actor.state_dict(), f'{path}/actor.params')
        torch.save(self.actor_target.state_dict(), f'{path}/actor_target.params')
        torch.save(self.critic.state_dict(), f'{path}/critic.params')
        torch.save(self.critic_target.state_dict(), f'{path}/critic_target.params')

    def load(self, path):
        self.actor.load_state_dict(torch.load(f'{path}/actor.params', map_location=self.device))
        self.actor_target.load_state_dict(torch.load(f'{path}/actor_target.params', map_location=self.device))
        self.critic.load_state_dict(torch.load(f'{path}/critic.params', map_location=self.device))
        self.critic_target.load_state_dict(torch.load(f'{path}/critic_target.params', map_location=self.device))
        info_json = json.load(open(f'{path}/info.json', 'r'))

        self.epsilon = info_json['epsilon']
        self.gamma = info_json['gamma']
        self.batch_size = info_json['batch_size']
        self.epsilon_decrement = info_json['epsilon_decrement']

        return int(info_json['episode'])





