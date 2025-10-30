"""Quantum encoder module for the UnifiedAutoencoder.

This module implements a variational quantum encoder that maps a
classical latent vector to a quantum‑derived representation.  The
circuit is built from a stack of RealAmplitudes layers interleaved
with CZ gates.  The output is the expectation value of Pauli‑Z on
each qubit, producing a real‑valued vector that can be fed into a
classical decoder.
"""

import torch
import torchquantum as tq
from torch import nn


class QuantumEncoder(tq.QuantumModule):
    """Variational quantum encoder.

    The circuit maps a classical latent vector `x` of shape
    (batch, latent_dim) to a set of rotation angles that drive a
    RealAmplitudes layer.  The circuit is repeated `reps` times
    and interleaved with CZ gates to introduce entanglement.
    """

    def __init__(
        self,
        latent_dim: int,
        qreg_size: int = 4,
        reps: int = 5,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.qreg_size = qreg_size
        self.reps = reps

        # Map the classical latent vector to rotation angles for the
        # RealAmplitudes layer.  The output dimension of the linear
        # layer is the number of qubits in the register.
        self.angle_mapper = nn.Linear(latent_dim, qreg_size)

        # Build the variational circuit.
        self.circuit = tq.QuantumCircuit(qreg_size)
        for _ in range(reps):
            self.circuit.add_layer(tq.RealAmplitudes(qreg_size))
            self.circuit.add_layer(tq.CZGate(qreg_size))

        # Measure all qubits in the Pauli‑Z basis.
        self.measure = tq.MeasureAll(tq.PauliZ)

        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, latent_dim)

        Returns:
            Tensor of shape (batch, qreg_size) containing the
            expectation values of Pauli‑Z on each qubit.
        """
        angles = self.angle_mapper(x)  # (batch, qreg_size)
        qdev = tq.QuantumDevice(self.qreg_size, bsz=x.shape[0], device=x.device)
        # Apply the variational circuit with the generated angles.
        self.circuit(qdev, angles)
        out = self.measure(qdev)  # (batch, qreg_size)
        return out


__all__ = ["QuantumEncoder"]
