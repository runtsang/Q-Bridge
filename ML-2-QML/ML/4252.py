"""Hybrid classical-quantum estimator combining QCNN feature extractor and RBF kernel.

The class :class:`HybridEstimatorQNN` exposes a classical feature extractor and
an optional quantum estimator.  It can be used as a drop‑in replacement for
the original EstimatorQNN while providing richer feature engineering
through a convolution‑inspired network and kernel‑based weighting.

The implementation keeps the interface lightweight: the constructor accepts
an optional ``quantum_estimator`` which must expose a ``predict`` method
compatible with scikit‑learn.  The forward pass returns the classical
features; the ``predict_quantum`` helper forwards the input to the quantum
backend when available.
"""

import torch
from torch import nn

# --------------------------------------------------------------------------- #
# Classical QCNN feature extractor
# --------------------------------------------------------------------------- #
class QCNNModel(nn.Module):
    """
    Stack of fully connected layers emulating the quantum convolution steps.
    """
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

# --------------------------------------------------------------------------- #
# RBF kernel module
# --------------------------------------------------------------------------- #
class RBFKernel(nn.Module):
    """
    Radial-basis-function kernel for classical weighting.
    """
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

# --------------------------------------------------------------------------- #
# Hybrid estimator
# --------------------------------------------------------------------------- #
class HybridEstimatorQNN(nn.Module):
    """
    Hybrid estimator that first processes the input with a QCNN feature extractor
    and optionally delegates the final prediction to a quantum EstimatorQNN.
    """
    def __init__(self, quantum_estimator=None, gamma: float = 1.0) -> None:
        super().__init__()
        self.feature_extractor = QCNNModel()
        self.kernel = RBFKernel(gamma)
        self.quantum_estimator = quantum_estimator

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Return the extracted classical features.  The output can be used for
        downstream classical regression or classification.
        """
        return self.feature_extractor(inputs)

    def predict_quantum(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward the input to the quantum backend and return predictions as a
        torch tensor.  The ``quantum_estimator`` must expose a ``predict``
        method compatible with scikit‑learn.
        """
        if self.quantum_estimator is None:
            raise RuntimeError("Quantum estimator not supplied.")
        # Convert to NumPy for the qiskit estimator
        np_inputs = inputs.detach().cpu().numpy()
        preds = self.quantum_estimator.predict(np_inputs)
        return torch.tensor(preds, dtype=inputs.dtype, device=inputs.device)

__all__ = ["HybridEstimatorQNN", "QCNNModel", "RBFKernel"]
