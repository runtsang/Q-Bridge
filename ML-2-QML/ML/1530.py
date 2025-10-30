import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QCNet(nn.Module):
    """
    Classical binary classifier that mimics the structure of the original hybrid model.
    The network consists of two convolutional blocks followed by three fully‑connected
    layers, and a lightweight fully‑connected head that produces two class scores
    via a softmax.  A sigmoid‑based activation is applied to the first head
    output to emulate the quantum expectation layer, and the final output is
    normalised to a probability distribution.
    """

    def __init__(self) -> None:
        super().__init__()
        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully‑connected backbone
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Classical head that replaces the quantum layer
        self.prob_head = nn.Linear(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Emulate the quantum expectation: sigmoid + shift
        logits = torch.sigmoid(x)
        logits = self.prob_head(logits)
        probs = F.softmax(logits, dim=-1)
        return probs

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience inference method that returns class probabilities.
        """
        self.eval()
        with torch.no_grad():
            return self(x)

__all__ = ["QCNet"]
