"""Classical hybrid model for binary classification with a CNN backbone and a feed‑forward classifier head.

The implementation is inspired by the original QCNet but replaces the quantum circuit
with a classical multilayer perceptron built by `build_classifier_circuit`.  The
function returns a `nn.Sequential` that maps a single scalar input to a two‑class
logit vector, mimicking the quantum expectation head.

In addition a lightweight `FastEstimator` utility is provided to evaluate the
model on a set of parameter vectors, optionally adding Gaussian shot noise
to emulate quantum measurement statistics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable, Sequence, List, Callable

# --------------------------------------------------------------------------- #
# Utility: build_classifier_circuit (classical variant)
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_features: int, depth: int) -> tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """Construct a feed‑forward classifier mirroring the quantum helper interface.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input to the classifier head.
    depth : int
        Number of hidden layers.

    Returns
    -------
    network : nn.Module
        Sequential network producing logits for two classes.
    encoding : Iterable[int]
        Dummy encoding indices (kept for API compatibility).
    weight_sizes : Iterable[int]
        Number of trainable parameters per layer.
    observables : List[int]
        Dummy observable indices (kept for API compatibility).
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

# --------------------------------------------------------------------------- #
# Estimator utilities
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """Evaluate a neural network for a batch of inputs and scalar observables."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
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
    """Adds optional Gaussian shot noise to deterministic predictions."""

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

# --------------------------------------------------------------------------- #
# HybridQCNet (classical)
# --------------------------------------------------------------------------- #
class HybridQCNet(nn.Module):
    """CNN backbone followed by a classical classifier head.

    The architecture mirrors the original QCNet but replaces the quantum expectation
    head with a feed‑forward network produced by :func:`build_classifier_circuit`.
    """

    def __init__(self, num_features: int = 3, depth: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Classical classifier head
        self.classifier, _, _, _ = build_classifier_circuit(num_features=1, depth=depth)
        # The head produces logits for two classes; we keep them raw for downstream
        # loss functions such as BCEWithLogitsLoss.

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

__all__ = ["HybridQCNet", "FastEstimator", "FastBaseEstimator", "build_classifier_circuit"]
