"""Quantum regularizer for the hybrid autoencoder.

The regularizer implements a simple variational circuit that
evaluates an expectation value of a PauliZ observable after
encoding the latent vector.  The expectation value is
interpreted as a fidelity‑like penalty and is returned as a
scalar tensor that can be added to any classical loss.

The circuit is built with PennyLane and uses the Torch interface
so it can be called directly from a PyTorch training loop.
"""

from __future__ import annotations

import pennylane as qml
import torch
from typing import Callable


class QuantumRegularizer:
    """Variational quantum circuit that returns a regularization score."""

    def __init__(self, device: str = "default.qubit", wires: int | None = None) -> None:
        """
        Parameters
        ----------
        device : str
            PennyLane device name.  Use ``"default.qubit"`` for fast CPU simulation
            or a Qiskit backend for real hardware.
        wires : int | None
            Number of qubits.  If None, the number of qubits will be inferred
            from the first latent vector passed to the circuit.
        """
        self.device_name = device
        self.wires = wires
        self.qnode: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None

    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantum penalty for a batch of latent vectors.

        Parameters
        ----------
        latent : torch.Tensor
            Tensor of shape (batch, latent_dim).  The latent vector is
            encoded into the first ``latent_dim`` qubits via rotations.
        Returns
        -------
        torch.Tensor
            Penalty of shape (batch,).  Lower values correspond to
            higher fidelity with the reference state.
        """
        batch, latent_dim = latent.shape
        if self.wires is None:
            self.wires = latent_dim
        if self.qnode is None:
            self.qnode = self._build_qnode(self.wires)

        # The QNode expects a 1‑D vector for parameters.
        # We flatten the batch to call the QNode efficiently.
        flat = latent.reshape(-1, latent_dim)
        # Execute the QNode; the result has shape (batch * latent_dim,)
        raw = self.qnode(flat)  # shape: (batch * latent_dim,)
        # Reshape back to (batch, latent_dim) and average over qubits
        return raw.reshape(batch, latent_dim).mean(dim=1)

    def _build_qnode(self, wires: int) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        dev = qml.device(self.device_name, wires=wires)

        @qml.qnode(dev, interface="torch")
        def circuit(params: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            # Encode latent vector via rotation around X
            for i in range(wires):
                qml.RX(params[i], wires=i)

            # Variational block
            for i in range(wires - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.Layer(qml.BasicEntanglerLayer, wires=range(wires), weights=weights)

            # Observe PauliZ on all qubits and sum
            return sum(qml.expval(qml.PauliZ(i)) for i in range(wires))

        # Initialise variational weights
        init_w = torch.randn(wires)
        # Return a lambda that injects weights at call time
        return lambda params: circuit(params, init_w)

__all__ = ["QuantumRegularizer"]
