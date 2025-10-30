import numpy as np
import torch
from torch import nn
from typing import Iterable, Sequence, Tuple

class ConvFilter(nn.Module):
    """Simple 2‑D convolutional filter mimicking a quanvolution layer."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: np.ndarray) -> float:
        tensor = torch.as_tensor(data, dtype=torch.float32).view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

class Kernel(nn.Module):
    """Radial‑basis‑function kernel with trainable gamma."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    layers = []
    in_dim = num_features
    for _ in range(depth):
        layers.append(nn.Linear(in_dim, num_features))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(num_features, 2))
    network = nn.Sequential(*layers)
    encoding = list(range(num_features))
    weight_sizes = [layers[i].weight.numel() + layers[i].bias.numel() for i in range(0, len(layers), 2)]
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

class HybridFCL:
    """Hybrid classical‑quantum fully‑connected layer with convolution, kernel, and classifier."""
    def __init__(self,
                 conv_kernel_size: int = 2,
                 linear_features: int = 1,
                 depth: int = 2,
                 kernel_gamma: float = 1.0,
                 num_classes: int = 2,
                 reference_points: int = 10) -> None:
        self.conv = ConvFilter(conv_kernel_size)
        self.linear = nn.Linear(1, linear_features)
        self.kernel = Kernel(kernel_gamma)
        self.ref_points = torch.randn(reference_points, linear_features)
        self.classifier, _, _, _ = build_classifier_circuit(reference_points, depth)

    def run(self, data: np.ndarray, thetas: Iterable[float] | None = None) -> np.ndarray:
        conv_out = self.conv.run(data)
        linear_in = torch.tensor([conv_out], dtype=torch.float32)
        linear_out = self.linear(linear_in).unsqueeze(0)  # shape (1, linear_features)
        kernel_vec = self.kernel(linear_out, self.ref_points)  # shape (1, reference_points)
        features = kernel_vec.squeeze()  # shape (reference_points,)
        logits = self.classifier(features.unsqueeze(0))  # shape (1, num_classes)
        probs = torch.softmax(logits, dim=-1).squeeze().detach().numpy()
        return probs

__all__ = ["HybridFCL"]
