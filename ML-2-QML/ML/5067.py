"""Unified hybrid network with classical and quantum sub‑modules.

This module implements four callables that can replace the seed files
Conv.py, EstimatorQNN.py, SamplerQNN.py, and ClassicalQuantumBinaryClassification.py.
Each callable returns a PyTorch module that mirrors the behaviour of its
quantum counterpart while adding classical refinements such as dropout,
sparsity, and adaptive thresholds.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 1. Classical Conv filter
# --------------------------------------------------------------------------- #
class _ConvFilter(nn.Module):
    """A PyTorch module that emulates a quanvolution layer with additional
    classical extensions such as dropout and a tunable activation threshold.
    """

    def __init__(self, kernel_size: int = 2, bias: bool = True,
                 dropout: float = 0.0, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size,
                              bias=bias, padding=kernel_size // 2)

    def run(self, data: np.ndarray | torch.Tensor) -> float:
        """Forward pass that returns a scalar mean activation."""
        tensor = torch.as_tensor(data, dtype=torch.float32)\
                   .view(1, 1, self.kernel_size, self.kernel_size)
        if self.dropout is not None:
            tensor = self.dropout(tensor)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

def Conv(**kwargs) -> _ConvFilter:
    """Factory returning a Conv instance."""
    return _ConvFilter(**kwargs)


# --------------------------------------------------------------------------- #
# 2. Classical EstimatorQNN
# --------------------------------------------------------------------------- #
class _EstimatorNN(nn.Module):
    """Small fully‑connected network that mimics a quantum estimator QNN."""

    def __init__(self, input_dim: int = 2,
                 hidden_dims: tuple[int, int] = (8, 4)) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Tanh(),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

def EstimatorQNN(**kwargs) -> _EstimatorNN:
    """Factory returning an EstimatorQNN instance."""
    return _EstimatorNN(**kwargs)


# --------------------------------------------------------------------------- #
# 3. Classical SamplerQNN
# --------------------------------------------------------------------------- #
class _SamplerModule(nn.Module):
    """Neural sampler that outputs a probability distribution over two classes."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)

def SamplerQNN(**kwargs) -> _SamplerModule:
    """Factory returning a SamplerQNN instance."""
    return _SamplerModule(**kwargs)


# --------------------------------------------------------------------------- #
# 4. Classical‑Quantum QCNet
# --------------------------------------------------------------------------- #
# Import the hybrid quantum layer lazily; if unavailable an error will be raised at runtime.
try:
    from.qml_code import Hybrid  # type: ignore[import]
except Exception:  # pragma: no cover
    Hybrid = None

class QCNet(nn.Module):
    """CNN followed by a quantum expectation head.

    The architecture mirrors the original hybrid model but replaces the
    quantum circuit with the ``Hybrid`` class defined in ``qml_code``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # For 32×32 RGB input images the feature map size after the
        # convolutional blocks is 15×8×8 → 960 elements.
        self.feature_dim = 960
        self.fc1 = nn.Linear(self.feature_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        if Hybrid is None:
            raise RuntimeError(
                "Hybrid quantum layer not found; ensure qml_code is importable."
            )
        import qiskit  # local import to avoid hard dependency
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(
            n_qubits=self.fc3.out_features,
            backend=backend,
            shots=100,
            shift=np.pi / 2,
        )

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
        x = self.hybrid(x).T
        return torch.cat((x, 1 - x), dim=-1)

__all__ = ["Conv", "EstimatorQNN", "SamplerQNN", "QCNet"]
