"""Quantum fully connected layer using TorchQuantum.

The quantum module implements a variational circuit that encodes
each feature into a separate qubit via angle‑encoding, applies a
randomised layer followed by trainable RX/RZ rotations, and finally
measures the Pauli‑Z expectation on each qubit. The output is
batch‑normalised to match the classical counterpart.
"""

from __future__ import annotations

import torch
from torch import nn
import torchquantum as tq
import torchquantum.functional as tqf


class FCL(tq.QuantumModule):
    """Quantum fully connected layer.

    Parameters
    ----------
    n_features : int, default=4
        Number of input features per sample. Must match the number of
        qubits (`n_wires`) for a one‑to‑one encoding.
    n_wires : int, default=4
        Number of qubits used in the circuit.
    """

    def __init__(self, n_features: int = 4, n_wires: int = 4) -> None:
        super().__init__()
        assert n_features == n_wires, "n_features must equal n_wires for one‑to‑one encoding."
        self.n_features = n_features
        self.n_wires = n_wires

        # Encoder: angle‑encoding with RY gates
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict["4x4_ry"]
        )

        # Randomised layer to inject expressivity
        self.random_layer = tq.RandomLayer(
            n_ops=30, wires=list(range(n_wires))
        )

        # Trainable rotations
        self.rx = tq.RX(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)

        # Measurement of Pauli‑Z on all wires
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Batch‑norm for output scaling
        self.norm = nn.BatchNorm1d(n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, n_features).

        Returns
        -------
        torch.Tensor
            Normalised quantum expectation values of shape (batch, n_wires).
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=True,
        )

        # Encode classical data
        self.encoder(qdev, x)

        # Variational circuit
        self.random_layer(qdev)
        self.rx(qdev)
        self.rz(qdev)

        # Measurement and scaling
        out = self.measure(qdev)
        return self.norm(out)

    def run(self, thetas: list[float]) -> torch.Tensor:  # pragma: no cover
        """Compatibility shim for the original seed interface."""
        return self.forward(torch.tensor(thetas))


def FCL_factory() -> FCL:
    """Convenience factory matching the original ``FCL`` function.

    Returns
    -------
    FCL
        An instance of the quantum fully connected layer.
    """
    return FCL()


__all__ = ["FCL", "FCL_factory"]
