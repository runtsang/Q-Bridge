import torch
from torch import nn
from typing import Iterable, Tuple

class HybridQCNN(nn.Module):
    """
    Classical neural network mirroring a QCNN architecture with a downstream
    classifier head. The network is built from linear layers that emulate the
    convolution and pooling operations of the quantum circuit, followed by a
    classifier network inspired by the incremental data‑uploading ansatz.
    """
    def __init__(self,
                 input_dim: int = 8,
                 conv_depth: int = 3,
                 classifier_depth: int = 2,
                 num_classes: int = 2) -> None:
        super().__init__()

        # Feature map – first “convolution” layer
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh())

        # Convolution + pooling pipeline (linear emulation of quantum QCNN)
        self.layers = nn.ModuleList()
        in_dim = 16
        for _ in range(conv_depth):
            conv = nn.Linear(in_dim, 16)
            pool = nn.Linear(16, 12)
            self.layers.extend([conv, nn.Tanh(), pool, nn.Tanh()])
            in_dim = 12

        # Classifier block (mirrors the classical build_classifier_circuit)
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 8), nn.ReLU(),
            nn.Linear(8, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        for layer in self.layers:
            x = layer(x)
        return torch.sigmoid(self.classifier(x))

def build_classifier_circuit(num_features: int,
                             depth: int) -> Tuple[nn.Module,
                                                  Iterable[int],
                                                  Iterable[int],
                                                  list[int]]:
    """
    Construct a feed‑forward classifier matching the quantum
    incremental data‑uploading interface.
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

__all__ = ["HybridQCNN", "build_classifier_circuit"]
