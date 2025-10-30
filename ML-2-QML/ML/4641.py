"""Combined classical QCNN and fast estimator utilities."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Callable, Iterable, List, Sequence

# --------- Classical QCNN ----------
class QCNNUnified(nn.Module):
    """Classical neural network mirroring the QCNN structure with dynamic sizing.

    The architecture follows the sequence of feature mapping, convolution, and pooling
    layers from the original QCNN prototype, but introduces a configurable hidden
    dimension for experiments on depth versus width trade‑offs.
    """
    def __init__(self, input_dim: int = 8, hidden_dim: int = 16) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(hidden_dim // 4, hidden_dim // 8), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(hidden_dim // 8, hidden_dim // 8), nn.Tanh())
        self.head = nn.Linear(hidden_dim // 8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

# --------- Classifier helper ----------
def build_classifier_circuit(
    num_features: int, depth: int
) -> tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """Return a feed‑forward classifier and metadata mirroring the quantum builder."""
    layers: list[nn.Module] = []
    in_dim = num_features
    weight_sizes: list[int] = []
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
    return network, list(range(num_features)), weight_sizes, observables

# --------- Estimators ----------
def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastBaseEstimator:
    """Batch evaluator for deterministic neural networks."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    """Adds Gaussian shot‑noise emulation to the deterministic evaluator."""
    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
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

# --------- Factory ----------
def QCNN() -> tuple[QCNNUnified, FastEstimator]:
    """Convenience constructor returning a classical QCNN model and its fast estimator."""
    model = QCNNUnified()
    estimator = FastEstimator(model)
    return model, estimator

__all__ = ["QCNNUnified", "QCNN", "build_classifier_circuit", "FastBaseEstimator", "FastEstimator"]
