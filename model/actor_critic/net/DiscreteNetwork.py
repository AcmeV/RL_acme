import torch.nn.functional as F
from torch import nn

class DiscreteNetwork(nn.Module):

    def __init__(self, n_features, n_hiddens, n_actions, activate=False):

        super(DiscreteNetwork, self).__init__()

        self.activate = activate

        self.net = nn.Sequential(
            nn.Linear(n_features, n_hiddens),
            nn.ReLU(),
            nn.Linear(n_hiddens, n_hiddens),
            nn.ReLU(),
            nn.Linear(n_hiddens, n_hiddens),
            nn.ReLU(),
            nn.Linear(n_hiddens, n_actions))

    def forward(self, X):

        if self.activate:
            return F.softmax(self.net(X))
        else:
            return self.net(X)