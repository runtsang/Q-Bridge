"""Extended classical binary classifier with interchangeable heads.

This module defines QuantumHybridBinaryClassifier that can be used with a
classical sigmoid head. The architecture is a lightweight CNN followed by
a dense head. It also exposes a `feature_extractor` method that returns the
flattened features before the head, enabling downstream use.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumHybridBinaryClassifier(nn.Module):
    """CNN backbone + classical sigmoid head for binary classification."""
    def __init__(self,
                 in_channels: int = 3,
                 conv_channels: list | None = None,
                 hidden_dims: list[int] = [120, 84],
                 dropout_probs: tuple[float, float] = (0.2, 0.5),
                 shift: float = 0.0):
        super().__init__()
        if conv_channels is None:
            conv_channels = [6, 15]
        self.conv1 = nn.Conv2d(in_channels, conv_channels[0], kernel_size=5,
                               stride=2, padding=1)
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3,
                               stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=dropout_probs[0])
        self.drop2 = nn.Dropout2d(p=dropout_probs[1])
        # Compute the flattened feature size after conv layers
        dummy_input = torch.zeros(1, in_channels, 32, 32)
        with torch.no_grad():
            x = self._forward_conv(dummy_input)
            flat_size = x.view(1, -1).size(1)
        self.fc1 = nn.Linear(flat_size, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 1)
        self.shift = shift

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        probs = torch.sigmoid(logits + self.shift)
        return torch.cat((probs, 1 - probs), dim=-1)

    def feature_extractor(self, x):
        """Return flattened features before the dense head."""
        x = self._forward_conv(x)
        return torch.flatten(x, 1)

__all__ = ["QuantumHybridBinaryClassifier"]
