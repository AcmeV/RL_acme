
from torch import nn

class ContinueNetwork(nn.Module):

    def __init__(self, n_features, n_hiddens):

        super(ContinueNetwork, self).__init__()

        self.pre_net = nn.Sequential(
            nn.Linear(n_features, n_hiddens),
            nn.ReLU(),
            nn.Linear(n_hiddens, n_hiddens),
            nn.ReLU(),
            nn.Linear(n_hiddens, n_hiddens),
            nn.ReLU())

        self.mu = nn.Sequential(
            nn.Linear(n_hiddens, 1),
            nn.Tanh()
        )

        self.gamma = nn.Sequential(
            nn.Linear(n_hiddens, 1),
            nn.Softplus()
        )

    def forward(self, X):

        X = self.pre_net(X)

        return self.mu(X), self.gamma(X)