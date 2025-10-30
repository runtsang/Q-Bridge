import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Iterable, List

class HybridQCNet(nn.Module):
    """Classical convolutional classifier with a hybrid head that can be swapped with a quantum head."""
    def __init__(self, use_quantum: bool = False):
        super().__init__()
        self.use_quantum = use_quantum
        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        # head
        self.head = nn.Linear(1, 1) if use_quantum else nn.Sigmoid()

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
        if self.use_quantum:
            logits = self.head(x)
        else:
            logits = torch.sigmoid(x)
        return torch.cat((logits, 1 - logits), dim=-1)

    def evaluate(self,
                 observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
                 parameter_sets: List[List[float]]) -> List[List[float]]:
        """Evaluate the model for a list of parameter sets and observables.
        Mimics FastBaseEstimator: for each set of parameters, compute the output and apply observables."""
        results = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                # treat params as additional inputs (e.g., biases) - here we ignore them for simplicity
                out = self.forward(torch.tensor(params).unsqueeze(0))
                row = []
                for obs in observables:
                    val = obs(out)
                    row.append(float(val.mean().item()) if isinstance(val, torch.Tensor) else float(val))
                results.append(row)
        return results
