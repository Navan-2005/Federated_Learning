import torch.nn as nn
import torch.nn.functional as F


class ANN_Model(nn.Module):
    def __init__(self, input_features=784, hidden1=128, hidden2=64, out_features=10):
        super().__init__()
        self.f_connected1 = nn.Linear(input_features, hidden1)
        self.f_connected2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, out_features)

    def forward(self, x):
        x = F.relu(self.f_connected1(x))
        x = F.relu(self.f_connected2(x))
        return self.out(x)
