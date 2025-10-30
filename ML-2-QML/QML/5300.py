"""Quantum estimator that uses a variational circuit on an encoded state.

The implementation is fully quantum‑centric.  It mirrors the EstimatorQNN
interface but replaces the classical feed‑forward stack with a Pennylane
variational circuit and a classical linear head.  The circuit accepts
classically‑encoded features as rotation angles and applies a
strongly‑entangling layer.  The output is a single expectation value
which is passed through a linear head to produce a regression score.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import pennylane as qml

# --------------------------------------------------------------------------- #
#  Utility: encode classical features into quantum rotations
# --------------------------------------------------------------------------- #

def _parametric_rotation_circuit(params: np.ndarray, wires: list[int]) -> None:
    """Apply a rotation RY to each wire with the corresponding parameter."""
    for w, p in zip(wires, params):
        qml.RY(p, wires=w)

# --------------------------------------------------------------------------- #
#  Quantum model
# --------------------------------------------------------------------------- #

class UnifiedEstimatorQNN(nn.Module):
    """Quantum estimator that maps classical inputs to a scalar output.

    The network consists of:
      * a parameterized rotation layer that encodes the input features
      * a strongly‑entangling variational layer (Pennylane's ``StronglyEntanglingLayers``)
      * a measurement of the Pauli‑Z expectation on all wires
      * a classical linear head that maps the measurement vector to a scalar
    """
    def __init__(self, num_wires: int, n_layers: int = 2, n_qubits: int = 4):
        super().__init__()
        self.num_wires = num_wires
        self.device = qml.device("default.qubit", wires=num_wires, shots=None)

        # Variational parameters for the entangling layers
        self.var_params = nn.Parameter(
            torch.randn(n_layers, num_wires, 3, dtype=torch.float32)
        )

        # Classical linear head
        self.head = nn.Linear(num_wires, 1)

        # Build the QNode
        @qml.qnode(self.device, interface="torch")
        def circuit(x: torch.Tensor, var: torch.Tensor):
            # Encode the classical input as RY rotations
            for i in range(num_wires):
                qml.RY(x[i], wires=i)
            # Apply strongly entangling layers
            qml.StronglyEntanglingLayers(var, wires=range(num_wires))
            # Measure all wires in Z basis
            return [qml.expval(qml.PauliZ(i)) for i in range(num_wires)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the quantum estimator.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, num_wires).  Each element is interpreted as an
            angle for an RY rotation on the corresponding wire.
        """
        # Broadcast per‑batch: the QNode expects 1‑D input per call
        batch_outputs = []
        for sample in x.unbind(dim=0):
            # Convert to torch tensor on same device
            sample = sample.to(torch.float32)
            q_out = self.circuit(sample, self.var_params)
            batch_outputs.append(q_out)
        q_out = torch.stack(batch_outputs, dim=0)
        return self.head(q_out).squeeze(-1)

# --------------------------------------------------------------------------- #
#  Factory function (mirrors original EstimatorQNN API)
# --------------------------------------------------------------------------- #

def EstimatorQNN(num_wires: int = 4, n_layers: int = 2, n_qubits: int = 4) -> UnifiedEstimatorQNN:
    """Return a quantum estimator with the specified number of wires."""
    return UnifiedEstimatorQNN(num_wires=num_wires, n_layers=n_layers, n_qubits=n_qubits)

__all__ = ["EstimatorQNN", "UnifiedEstimatorQNN"]
