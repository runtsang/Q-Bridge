import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QCNet(nn.Module):
    """Classical CNN with residual MLP head and batch norm, producing binary probabilities."""
    def __init__(self):
        super().__init__()
        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(15)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop_conv = nn.Dropout2d(p=0.3)

        # Fully connected head with residual connection
        self.fc1 = nn.Linear(540, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.res_fc = nn.Linear(120, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional backbone
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.drop_conv(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop_conv(x)

        # Flatten and MLP head
        x = torch.flatten(x, 1)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        out = self.fc3(h2)

        # Residual from first hidden layer
        res = self.res_fc(h1)
        out = out + res

        # Sigmoid activation to produce probability
        prob = torch.sigmoid(out)
        return torch.cat((prob, 1 - prob), dim=-1)

__all__ = ["QCNet"]
