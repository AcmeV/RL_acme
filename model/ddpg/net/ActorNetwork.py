import torch
from torch import nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(ActorNetwork, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, s):
        x = F.relu(self.linear1(s))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x