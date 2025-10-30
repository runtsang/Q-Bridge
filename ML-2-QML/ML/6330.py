"""Classical binary classifier with a hybrid dense head and fast evaluation utilities.

This module defines a lightweight PyTorch network that mirrors the
convolutional backbone of the quantum model.  The final head is a
parameterised dense layer wrapped in a differentiable sigmoid
function.  A FastEstimator class is provided for quick batch
evaluation and optional Gaussian shot noise, mirroring the
FastBaseEstimator from the original repository.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation that mimics the quantum expectation head."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:  # type: ignore[override]
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class Hybrid(nn.Module):
    """Simple dense head that replaces the quantum circuit in the original model."""

    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)

class HybridBinaryClassifier(nn.Module):
    """Convolutional network followed by a classical dense head."""

    def __init__(self,
                 conv_filters: Optional[Sequence[int]] = None,
                 fc_sizes: Optional[Sequence[int]] = None) -> None:
        super().__init__()
        conv_filters = conv_filters or [6, 15]
        fc_sizes = fc_sizes or [120, 84]
        self.conv1 = nn.Conv2d(3, conv_filters[0], kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(conv_filters[0], conv_filters[1], kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        dummy_input = torch.zeros(1, 3, 252, 252)
        with torch.no_grad():
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.drop1(x)
            x = self.pool(F.relu(self.conv2(x)))
            x = self.drop1(x)
            x = torch.flatten(x, 1)
        flat_features = x.shape[1]
        self.fc1 = nn.Linear(flat_features, fc_sizes[0])
        self.fc2 = nn.Linear(fc_sizes[0], fc_sizes[1])
        self.fc3 = nn.Linear(fc_sizes[1], 1)
        self.hybrid = Hybrid(1, shift=0.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
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
        probabilities = self.hybrid(x)
        return torch.cat((probabilities, 1 - probabilities), dim=-1)

class FastEstimator:
    """Fast evaluation of a PyTorch model with optional Gaussian shot noise.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Evaluate the model for a collection of parameter sets.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the model output and returns a scalar
            or a tensor that can be reduced to a scalar.
        parameter_sets : sequence of parameter vectors
            Each vector is fed as input to the model.
        shots : int, optional
            If supplied, Gaussian noise with variance 1/shots is added to
            each observable value.
        seed : int, optional
            Random seed for reproducible noise.
        """
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

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["HybridFunction", "Hybrid", "HybridBinaryClassifier", "FastEstimator"]
