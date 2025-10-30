import torch
import torch.nn as nn
from typing import Iterable, Tuple

class HybridClassifier(nn.Module):
    """
    Classical hybrid classifier that emulates the quantum helper interface.
    The model first applies a 2‑D convolution (Conv2d) to the input image
    and then passes the flattened feature vector through a depth‑controlled
    feed‑forward network.  The public API mirrors the quantum build
    function so that classical and quantum experiments can share the same
    experiment skeleton.
    """
    def __init__(self, num_features: int, depth: int, kernel_size: int = 3, threshold: float = 0.0):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.threshold = threshold
        self.classifier = self._build_classifier(num_features, depth)

    def _build_classifier(self, num_features: int, depth: int) -> nn.Sequential:
        layers = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        layers.append(nn.Linear(in_dim, 2))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, H, W)
        feat = self.conv(x)
        feat = torch.sigmoid(feat - self.threshold)
        flat = feat.view(feat.size(0), -1)
        return self.classifier(flat)

    @staticmethod
    def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
        """
        Mimic the quantum helper signature.  Returns the classifier module,
        an encoding list (indices of input features), the weight size list
        for each linear layer, and the observable indices (output classes).
        """
        layers = []
        in_dim = num_features
        weight_sizes = []
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.append(linear)
            layers.append(nn.ReLU())
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = num_features
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())
        network = nn.Sequential(*layers)
        encoding = list(range(num_features))
        observables = list(range(2))
        return network, encoding, weight_sizes, observables

__all__ = ["HybridClassifier"]
