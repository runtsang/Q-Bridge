"""Classical hybrid classifier with optional quantum‑inspired fully‑connected layer.

This module implements a CNN backbone identical to the original seed,
and a head that can be a standard sigmoid or a lightweight FCL that
mimics a parameterised quantum circuit using a tanh activation and
mean aggregation. The design allows easy swapping of the head for
comparative studies while keeping the entire pipeline fully classical.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClassicalFCL(nn.Module):
    """Classical emulation of the fully‑connected quantum layer."""
    def __init__(self, in_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def run(self, thetas: np.ndarray) -> np.ndarray:
        values = torch.as_tensor(thetas, dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()


class QCNet(nn.Module):
    """CNN backbone with a configurable head."""
    def __init__(self, head: str = "classical", fcl_features: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        if head == "classical":
            self.head = nn.Sequential(
                nn.Linear(1, 1),
                nn.Sigmoid()
            )
        elif head == "fcl":
            self.fcl = ClassicalFCL(fcl_features)
            self.head = nn.Sequential(
                nn.Linear(1, 1),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unsupported head type: {head}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
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

        if hasattr(self, "fcl"):
            probs = torch.tensor(self.fcl.run(x.detach().cpu().numpy()), device=x.device)
        else:
            probs = self.head(x)

        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["QCNet", "ClassicalFCL"]
