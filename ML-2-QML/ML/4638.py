"""Hybrid estimator combining classical convolution, fully connected, and quantum regression.

The module defines ``HybridEstimatorQNN`` which processes raw inputs through a
classical convolutional front‑end (``Conv``), a fully‑connected layer
(``FCL``), and finally a quantum regression head implemented by
``HybridQuantumEstimator`` from the QML module.

Key design points
-----------------
* The classical sub‑network can be swapped for any PyTorch module.
* The quantum head is evaluated via Qiskit’s Estimator primitive and
  therefore runs on any supported backend (qasm or statevector).
* The interface stays identical to the original EstimatorQNN example
  while exposing a scalable hybrid pipeline.

The network returns a scalar tensor that can be used directly in a
regression loss such as MSE.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

# Classical helpers
from Conv import Conv
from FCL import FCL

# Quantum evaluator
from EstimatorQNN import HybridQuantumEstimator


class HybridEstimatorQNN(nn.Module):
    """
    PyTorch module that fuses a classical front‑end with a quantum regression head.

    Parameters
    ----------
    conv_kernel : int, default=2
        Size of the convolution kernel used by the ``Conv`` filter.
    conv_threshold : float, default=0.0
        Threshold for the sigmoid activation in the convolutional filter.
    fcl_features : int, default=1
        Number of input features to the fully‑connected layer.
    quantum_n_qubits : int, default=1
        Number of qubits in the quantum regression circuit.
    """

    def __init__(
        self,
        conv_kernel: int = 2,
        conv_threshold: float = 0.0,
        fcl_features: int = 1,
        quantum_n_qubits: int = 1,
    ) -> None:
        super().__init__()
        self.conv = Conv()
        self.fcl = FCL()
        self.quantum = HybridQuantumEstimator(quantum_n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the hybrid network.

        Parameters
        ----------
        x : torch.Tensor
            Input data.  If it is a 4‑D tensor, it is interpreted as a batch of
            single‑channel images and passed to the convolutional filter.
            Otherwise it is treated as a flat vector and bypasses the front‑end.

        Returns
        -------
        torch.Tensor
            A 0‑D tensor containing the quantum‑evaluated regression output.
        """
        # Classical front‑end
        with torch.no_grad():
            if x.ndim == 4:  # batch of images
                batch_size = x.shape[0]
                conv_out = []
                for i in range(batch_size):
                    conv_out.append(self.conv.run(x[i].cpu().numpy()))
                conv_out = np.stack(conv_out)
            else:
                conv_out = self.conv.run(x.cpu().numpy())

        # Fully‑connected layer
        with torch.no_grad():
            fcl_out = self.fcl.run(conv_out)

        # Quantum regression head
        quantum_out = self.quantum.run(
            input_params=[0.0],  # placeholder for compatibility
            weight_params=[float(fcl_out)],
        )

        return torch.tensor(quantum_out, dtype=torch.float32)


__all__ = ["HybridEstimatorQNN"]
