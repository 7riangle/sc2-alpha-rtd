# alpha_model.py
# ResNet-style policy/value network

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        return F.relu(x + identity)


class ImprovedNNet(nn.Module):
    def __init__(self, state_size, action_size, width=512, n_blocks=3):
        super().__init__()
        self.inp = nn.Linear(state_size, width)
        self.bn_inp = nn.BatchNorm1d(width)
        self.blocks = nn.ModuleList([ResidualBlock(width) for _ in range(n_blocks)])
        self.policy = nn.Linear(width, action_size)
        self.value = nn.Linear(width, 1)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = F.relu(self.bn_inp(self.inp(x)))
        for blk in self.blocks:
            x = blk(x)
        logits = self.policy(x)
        value = torch.tanh(self.value(x))
        return logits, value