"""
Hybrid estimator that integrates a quantum convolution filter with a classical feed‑forward network.
The module is fully classical (PyTorch), but it lazily imports the quantum filter when the EstimatorQNN() factory is called.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

class HybridEstimator(nn.Module):
    """
    Classical regressor that augments the input with a quantum‑derived feature.
    """
    def __init__(self, quantum_filter, hidden_sizes=(8, 4)):
        """
        Parameters
        ----------
        quantum_filter : object
            Instance providing a `run(data: np.ndarray) -> float` method.
        hidden_sizes : tuple[int, int], optional
            Sizes of the hidden layers.
        """
        super().__init__()
        self.quantum_filter = quantum_filter
        self.classical_net = nn.Sequential(
            nn.Linear(3, hidden_sizes[0]),   # 2 raw inputs + 1 quantum feature
            nn.Tanh(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[1], 1),
        )

    def forward(self, inputs: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape (batch, 2) – two classical features.
        patch : torch.Tensor
            Shape (batch, kernel_size, kernel_size) – input for the quantum filter.

        Returns
        -------
        torch.Tensor
            Shape (batch, 1) – regression output.
        """
        # Compute quantum feature for each sample
        quantum_vals = []
        for i in range(inputs.shape[0]):
            # Convert patch to numpy for the quantum filter
            quantum_vals.append(self.quantum_filter.run(patch[i].cpu().numpy()))
        quantum_tensor = torch.tensor(quantum_vals, dtype=inputs.dtype, device=inputs.device).unsqueeze(1)

        # Concatenate classical and quantum features
        x = torch.cat([inputs, quantum_tensor], dim=1)
        return self.classical_net(x)

def EstimatorQNN() -> HybridEstimator:
    """
    Factory that returns a HybridEstimator using the default quantum filter.
    The quantum filter is imported lazily to avoid unnecessary Qiskit dependencies
    when the module is used purely classically.
    """
    try:
        from.qml_estimatorQNN import QuanvCircuit
    except ImportError as exc:
        raise ImportError(
            "Quantum filter module (qml_estimatorQNN) not found. "
            "Make sure the QML side is installed and importable."
        ) from exc

    quantum_filter = QuanvCircuit(kernel_size=2, threshold=0.5, shots=200)
    return HybridEstimator(quantum_filter)

__all__ = ["HybridEstimator", "EstimatorQNN"]
