"""Quantum helper for the hybrid autoencoder.

Provides a Pennylane‑based variational circuit that refines a latent vector.
The circuit is wrapped as a torch.nn.Module so it can be plugged into the
classical model defined in the ML module.

The circuit uses a simple RY encoding of the input followed by a few
parameterised layers of RZ and CNOT gates.  The output is the expectation
values of Pauli-Z on each qubit, forming a refined latent vector.
"""

from __future__ import annotations

import pennylane as qml
import pennylane.numpy as pnp
import torch
from torch import nn
from typing import Optional

def quantum_refiner_module(
    latent_dim: int,
    qnum: Optional[int] = None,
    reps: int = 2,
    device: str = "default.qubit",
) -> nn.Module:
    """Return a torch.nn.Module that refines a latent vector via a quantum circuit.

    Parameters
    ----------
    latent_dim
        Size of the input latent vector.
    qnum
        Number of qubits used in the circuit.  Defaults to ``latent_dim``.
    reps
        Number of variational layers.
    device
        Pennylane device string (e.g. ``default.qubit`` or ``qiskit_aer``).
    """
    qnum = qnum if qnum is not None else latent_dim
    dev = qml.device(device, wires=qnum)

    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def circuit(x: torch.Tensor) -> torch.Tensor:
        # Encode the classical vector into qubit rotations
        for i, val in enumerate(x):
            qml.RY(val, wires=i)
        # Variational layers
        for _ in range(reps):
            for i in range(qnum):
                qml.RZ(pnp.random.rand(), wires=i)
            for i in range(qnum - 1):
                qml.CNOT(wires=[i, i + 1])
        # Readout: Pauli‑Z expectation for each qubit
        return torch.stack([qml.expval(qml.PauliZ(i)) for i in range(qnum)])

    class QuantumRefinerModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.circuit = circuit

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Expect shape (batch, latent_dim)
            out = []
            for vec in x:
                out.append(self.circuit(vec))
            return torch.stack(out)

    return QuantumRefinerModule()

__all__ = ["quantum_refiner_module"]
