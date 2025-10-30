import torch
import torch.nn as nn
import torch.nn.functional as F

def build_classifier_circuit(num_features: int, depth: int):
    """Construct a feed‑forward classifier mirroring the quantum helper interface."""
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

class HybridQuantumClassifier(nn.Module):
    """
    Hybrid classical‑quantum classifier with optional quanvolution front‑end.
    """
    def __init__(self, num_features: int, depth: int = 2, use_quanvolution: bool = False):
        super().__init__()
        self.use_quanvolution = use_quanvolution
        if use_quanvolution:
            # Lightweight quanvolution inspired filter
            self.qfilter = nn.Conv2d(1, 4, kernel_size=2, stride=2)
            self.linear = nn.Linear(4 * 14 * 14, 10)
        else:
            self.network, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(num_features, depth)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_quanvolution:
            features = self.qfilter(x)
            logits = self.linear(features.view(x.size(0), -1))
            probs = F.softmax(logits, dim=-1)
            return probs
        else:
            logits = self.network(x)
            probs = F.softmax(logits, dim=-1)
            return probs

__all__ = ["HybridQuantumClassifier"]
