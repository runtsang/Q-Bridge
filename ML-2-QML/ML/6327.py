import torch
import torch.nn as nn
from typing import Tuple, Iterable, List

__all__ = ["HybridNATModel"]


class HybridNATModel(nn.Module):
    """
    Classical hybrid model: CNN backbone + quantum‑inspired fully connected head.
    Mirrors the QuantumNAT architecture but replaces the quantum module with
    a classical variational block that emulates a quantum feature map.
    """

    def __init__(
        self,
        conv_features: int = 16,
        fc_dim: int = 64,
        num_classes: int = 4,
        depth: int = 3,
    ) -> None:
        super().__init__()
        # Convolutional feature extractor
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, conv_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Classical variational head
        layers = [nn.Linear(conv_features * 7 * 7, fc_dim), nn.ReLU(inplace=True)]
        for _ in range(depth - 1):
            layers.append(nn.Linear(fc_dim, fc_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(fc_dim, num_classes))
        self.head = nn.Sequential(*layers)
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        flat = feats.view(feats.size(0), -1)
        out = self.head(flat)
        return self.norm(out)

    @staticmethod
    def build_classifier_circuit(
        num_features: int, depth: int
    ) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """
        Classical replacement for the quantum classifier factory.
        Returns a network, encoding indices, weight sizes and observables.
        """
        layers = []
        in_dim = num_features
        encoding = list(range(num_features))
        weight_sizes = []
        for _ in range(depth):
            lin = nn.Linear(in_dim, num_features)
            layers.append(lin)
            layers.append(nn.ReLU(inplace=True))
            weight_sizes.append(lin.weight.numel() + lin.bias.numel())
            in_dim = num_features
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())
        net = nn.Sequential(*layers)
        observables = list(range(2))
        return net, encoding, weight_sizes, observables
