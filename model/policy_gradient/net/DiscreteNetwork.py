
from torch import nn

class DiscreteNetwork(nn.Module):

    def __init__(self, n_features, n_hiddens, n_actions):

        super(DiscreteNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_features, n_hiddens),
            nn.ReLU(),
            nn.Linear(n_hiddens, n_hiddens),
            nn.ReLU(),
            nn.Linear(n_hiddens, n_hiddens),
            nn.ReLU(),
            nn.Linear(n_hiddens, n_actions),
            nn.Softmax()
        )

    def forward(self, X):
        return self.net(X)