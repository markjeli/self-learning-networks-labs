import torch.nn.functional as F
from torch import nn


class DQN(nn.Module):
    def __init__(self, num_features, num_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(num_features, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, num_actions)

    def forward(self, x):
        return self.layer3(F.relu(self.layer2(F.relu(self.layer1(x)))))
