"""Unified FastBaseEstimator for classical models with optional shot noise and helper factories.

The module extends the original lightweight estimator to:
* support Gaussian shot‑noise on the output
* expose a set of factory functions that mirror the quantum primitives
  (QCNN, FCL, QCNet) for easy comparison
* provide a ``Hybrid`` head that simulates a quantum expectation layer
  with a differentiable sigmoid.

The public API remains unchanged so existing code continues to run."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a list of floats to a 2‑D float tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model for batches of inputs and observables.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.  It must accept a 2‑D tensor of shape
        ``(batch, features)`` and return a 2‑D tensor.

    Notes
    -----
    The estimator is intentionally lightweight: it runs in ``eval`` mode and
    disables gradients.  For stochastic behaviour a Gaussian noise term can
    be added via the :py:meth:`evaluate` ``shots`` argument.
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
        """Return the value of each observable for each parameter set.

        The ``shots`` argument controls Gaussian noise added to each result;
        a value of ``None`` means deterministic evaluation.  ``seed`` is
        forwarded to :class:`numpy.random.Generator`.
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
                    scalar = float(value.mean().cpu()) if isinstance(value, torch.Tensor) else float(value)
                    row.append(scalar)
                results.append(row)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy = [[float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row] for row in results]
        return noisy


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        return super().evaluate(observables, parameter_sets, shots=shots, seed=seed)


# --------------------------------------------------------------------------- #
#  Classical helper factories
# --------------------------------------------------------------------------- #

class QCNNModel(nn.Module):
    """Stack of fully‑connected layers emulating a quantum convolutional network."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


def QCNN() -> QCNNModel:
    """Factory returning a pre‑configured QCNN model."""
    return QCNNModel()


class FullyConnectedLayer(nn.Module):
    """Simple fully‑connected layer used as a quantum‑style stand‑in."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()


def FCL() -> FullyConnectedLayer:
    """Return a layer mimicking the quantum fully‑connected block."""
    return FullyConnectedLayer()


class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation mimicking a quantum expectation head."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:  # type: ignore[override]
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None


class Hybrid(nn.Module):
    """Dense head that replaces a quantum circuit."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)


class QCNet(nn.Module):
    """CNN‑based binary classifier mirroring the hybrid quantum model."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(1, shift=0.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = torch.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.drop2(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        probabilities = self.hybrid(x)
        return torch.cat((probabilities, 1 - probabilities), dim=-1)


def QCNet() -> QCNet:
    """Return a pretrained QCNet classifier."""
    return QCNet()


__all__ = [
    "FastBaseEstimator",
    "FastEstimator",
    "QCNNModel",
    "QCNN",
    "FullyConnectedLayer",
    "FCL",
    "HybridFunction",
    "Hybrid",
    "QCNet",
]
