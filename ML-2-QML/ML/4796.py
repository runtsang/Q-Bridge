import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple, List

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """
    Construct a simple feed‑forward classifier that mirrors the quantum
    interface.  The network consists of `depth` hidden layers followed
    by a binary output head.  The function returns the network, a
    dummy encoding list, weight sizes, and a dummy observable list.
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding: List[int] = list(range(num_features))
    weight_sizes: List[int] = []

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
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


class QuanvolutionFilter(nn.Module):
    """
    Classical 2×2 convolutional filter that mimics the behaviour of a
    quanvolution layer but operates entirely in PyTorch.  The output
    is flattened into a 1‑D feature vector.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)


class ConvFilter(nn.Module):
    """
    Optional lightweight 2×2 convolution that can be used as a drop‑in
    replacement for the quanvolution filter.  It offers a threshold
    based activation that mimics the behaviour of the quantum filter.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data) -> float:
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()


class QuantumClassifierModel(nn.Module):
    """
    Classical classifier that can optionally prepend a convolutional
    filter before feeding the flattened features into a feed‑forward
    network.  The class name matches the quantum counterpart so that
    both modules expose a common API.
    """
    def __init__(
        self,
        num_features: int = 784,
        depth: int = 2,
        use_filter: bool = False,
        filter_type: str = "quanvolution",
    ) -> None:
        super().__init__()
        self.use_filter = use_filter

        if use_filter:
            if filter_type == "quanvolution":
                self.filter = QuanvolutionFilter()
                # The filter output has 4 channels and 14×14 patches
                num_features = 4 * 14 * 14
            elif filter_type == "conv":
                self.filter = ConvFilter()
                num_features = 1 * 14 * 14
            else:
                raise ValueError(f"Unknown filter type: {filter_type}")
        else:
            self.filter = None

        self.network, _, _, _ = build_classifier_circuit(num_features, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass through the optional filter followed by the
        classifier network.  Input `x` is expected to be a 4‑D tensor
        of shape (batch, 1, 28, 28) for image data.
        """
        if self.filter is not None:
            x = self.filter(x)
        return self.network(x)

__all__ = ["QuantumClassifierModel"]
