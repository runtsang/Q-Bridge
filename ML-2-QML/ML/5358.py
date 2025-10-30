from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def HybridSamplerQNN():
    class HybridSamplerQNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.feature_map = nn.Sequential(nn.Linear(2, 8), nn.Tanh())
            self.conv1 = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
            self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
            self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
            self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
            self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
            self.head = nn.Linear(4, 2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            x = self.feature_map(inputs)
            x = self.conv1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.pool2(x)
            x = self.conv3(x)
            return F.softmax(self.head(x), dim=-1)

    return HybridSamplerQNN()
