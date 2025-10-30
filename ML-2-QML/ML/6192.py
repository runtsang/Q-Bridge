import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple

def build_classifier_network(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """
    Construct a feed‑forward classifier mirroring the quantum depth.
    Returns the network, encoding indices, weight sizes, and observables.
    """
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

class HybridBinaryClassifier(nn.Module):
    """
    Classical convolutional network with a fully‑connected head that mimics the
    quantum expectation layer in the original hybrid model.
    The depth of the classifier head can be increased to match the depth of the
    quantum ansatz, enabling a fair comparison between the two paradigms.
    """
    def __init__(self, depth: int = 3, num_qubits: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_qubits)
        self.classifier, self.encoding, self.weight_sizes, self.observables = build_classifier_network(num_qubits, depth)

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
        logits = self.classifier(x)
        return logits

__all__ = ["HybridBinaryClassifier"]
