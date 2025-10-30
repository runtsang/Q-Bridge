import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuanvolutionFilter(nn.Module):
    """Classical 2×2 patch convolution mimicking a quantum kernel."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class FCL(nn.Module):
    """Placeholder for a fully‑connected quantum layer implemented classically."""
    def __init__(self, n_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: np.ndarray) -> np.ndarray:
        t = torch.as_tensor(thetas, dtype=torch.float32).view(-1, 1)
        out = torch.tanh(self.linear(t)).mean(dim=0)
        return out.detach().cpu().numpy()

class EstimatorQNN(nn.Module):
    """Simple regression head inspired by the quantum EstimatorQNN example."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class HybridQuantumClassifier(nn.Module):
    """Hybrid classifier that combines classical quanvolution, a linear head and optional quantum‑style layers."""
    def __init__(self, num_features: int = 28, depth: int = 2):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        # output size: 4 * 14 * 14 for 28×28 input
        self.fc = nn.Linear(4 * 14 * 14, 10)
        self.fcl = FCL()
        self.regressor = EstimatorQNN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.qfilter(x)
        logits = self.fc(feats)
        return F.log_softmax(logits, dim=-1)

    def run_fcl(self, thetas: np.ndarray) -> np.ndarray:
        """Run the classical surrogate of a fully‑connected quantum layer."""
        return self.fcl.run(thetas)

    def predict_regression(self, x: torch.Tensor) -> torch.Tensor:
        """Return regression output from the EstimatorQNN head."""
        return self.regressor(x)

def build_classifier_circuit(num_features: int, depth: int):
    """Construct a feed‑forward network that mirrors the quantum interface."""
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

__all__ = [
    "QuanvolutionFilter",
    "FCL",
    "EstimatorQNN",
    "HybridQuantumClassifier",
    "build_classifier_circuit",
]
