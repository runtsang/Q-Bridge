"""Hybrid Estimator combining classical convolution and quantum variational layer.

The module exposes a single function `EstimatorQNN` that returns a PyTorch
`nn.Module`.  The network consists of:

* A lightweight 2‑D convolutional filter (borrowed from the classical Conv
  seed) that extracts local features from the input image.
* A quantum variational layer implemented in the QML module.  The layer
  evaluates the expectation value of a Y‑Pauli operator on a parameterised
  circuit that encodes the input patch.  The QML module is imported lazily
  to keep the ML side free of quantum dependencies.
* A final linear regression head that maps the combined signal to a scalar
  output.

The design demonstrates how a classical pre‑processor can be coupled to a
quantum estimator to produce a hybrid model that can be trained end‑to‑end
with standard PyTorch optimisers.
"""

from __future__ import annotations

import torch
from torch import nn
import numpy as np

# Classical convolutional filter (from Conv.py)
class ConvFilter(nn.Module):
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply a single‑channel 2‑D convolution and sigmoid activation.
        """
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[2, 3])  # global average

# Lazy import of the quantum estimator to keep the ML side pure
def _load_quantum_estimator():
    from.quantum_estimator import HybridEstimator  # type: ignore
    return HybridEstimator

class HybridEstimator(nn.Module):
    """
    PyTorch implementation of the hybrid estimator.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        conv_threshold: float = 0.0,
        quantum_shots: int = 200,
        quantum_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size, conv_threshold)
        # The quantum estimator is instantiated lazily to avoid importing
        # Qiskit until the first forward pass.
        self._quantum_estimator_cls = _load_quantum_estimator()
        self.quantum_estimator = self._quantum_estimator_cls(
            num_qubits=kernel_size ** 2,
            shots=quantum_shots,
            threshold=quantum_threshold,
        )
        self.fc = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch, 1, H, W).

        Returns
        -------
        torch.Tensor
            Regression output of shape (batch, 1).
        """
        # Classical convolution output
        conv_out = self.conv(x)  # shape (batch, 1)

        # Quantum layer output
        # Prepare data for quantum estimator: flatten each patch
        batch_size = x.shape[0]
        quantum_out = []
        for i in range(batch_size):
            # Flatten the image to a 1‑D array of length kernel_size**2
            patch = x[i, 0, :, :].view(-1).cpu().numpy()
            # Pad or truncate to required length
            if patch.size < self.quantum_estimator.num_qubits:
                patch = np.pad(patch, (0, self.quantum_estimator.num_qubits - patch.size))
            else:
                patch = patch[: self.quantum_estimator.num_qubits]
            quantum_out.append(self.quantum_estimator.run(patch))
        quantum_out = torch.tensor(quantum_out, dtype=torch.float32, device=x.device).unsqueeze(1)

        # Combine classical and quantum signals
        combined = conv_out + quantum_out

        # Regression head
        out = self.fc(combined)
        return out

def EstimatorQNN() -> HybridEstimator:
    """
    Factory function compatible with the original EstimatorQNN anchor.
    Returns an instance of :class:`HybridEstimator`.
    """
    return HybridEstimator()
