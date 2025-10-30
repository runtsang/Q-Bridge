"""Hybrid quantum–classical classifier module.

This module implements a neural network that mirrors the structure of the
quantum circuit defined in the QML counterpart.  It combines:
  * a classical convolutional filter (ConvFilter) inspired by the quanvolution
    layer (Ref. 4),
  * a QCNN‑style stack of fully‑connected layers (QCNNModel) (Ref. 3),
  * a final linear head that produces a binary score,
  * utilities (FastBaseEstimator / FastEstimator) for batched evaluation.

The class :class:`HybridQuantumClassifier` can be instantiated with the
desired number of input features and depth.  It exposes the same
``build_classifier_circuit`` API as the quantum side, returning the network,
a list of feature indices, the weight sizes for each layer, and a list of
output observables (here simply the class indices).

"""

from __future__ import annotations

from typing import Callable, Iterable, List, Tuple

import torch
import torch.nn as nn
import numpy as np

# --------------------------------------------------------------------------- #
# 1. Classical building blocks
# --------------------------------------------------------------------------- #

class ConvFilter(nn.Module):
    """Convolutional filter that mimics the quanvolution layer (Ref. 4)."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tensor = x.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()


class QCNNModel(nn.Module):
    """QCNN‑style fully‑connected network (Ref. 3)."""

    def __init__(self, input_dim: int = 8) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


# --------------------------------------------------------------------------- #
# 2. Hybrid classifier
# --------------------------------------------------------------------------- #

class HybridQuantumClassifier(nn.Module):
    """Classical network that mirrors the quantum ansatz.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int
        Number of linear layers in the QCNN‑style stack.
    kernel_size : int
        Kernel size for the convolutional filter.
    """

    def __init__(self, num_features: int, depth: int = 3, kernel_size: int = 2) -> None:
        super().__init__()
        self.num_features = num_features
        self.depth = depth

        # 1. Data encoding
        self.encoder = nn.Linear(num_features, num_features)

        # 2. Convolutional filter
        self.conv_filter = ConvFilter(kernel_size=kernel_size)

        # 3. QCNN‑style stack
        self.qcnn = QCNNModel(input_dim=num_features)

        # 4. Final classifier head
        self.classifier = nn.Linear(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        filt = self.conv_filter(x)
        x = torch.cat([x, filt], dim=-1)
        x = self.qcnn(x)
        return self.classifier(x)

    def build_classifier_circuit(
        self,
    ) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """Return the module, feature indices, weight sizes, and observables."""
        encoding = list(range(self.num_features)) + [self.num_features]
        weight_sizes = []
        for layer in [self.encoder, self.conv_filter.conv, self.qcnn, self.classifier]:
            weight_sizes.append(sum(p.numel() for p in layer.parameters()))
        observables = [0, 1]
        return self, encoding, weight_sizes, observables


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """Convenience wrapper that creates a :class:`HybridQuantumClassifier`."""
    model = HybridQuantumClassifier(num_features, depth)
    return model.build_classifier_circuit()


# --------------------------------------------------------------------------- #
# 3. Estimator utilities
# --------------------------------------------------------------------------- #

def _ensure_batch(values: List[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model on batches of inputs."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: List[List[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: List[List[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = [
    "HybridQuantumClassifier",
    "build_classifier_circuit",
    "FastBaseEstimator",
    "FastEstimator",
    "ConvFilter",
    "QCNNModel",
]
