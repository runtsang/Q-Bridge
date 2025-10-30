"""Quantum module that consumes the 4‑D latent vector from the classical side.

The circuit is a lightweight variational layer built on top of a
general encoder and a RandomLayer, followed by a few parameterised
gates.  The output is a normalised vector of Pauli‑Z expectation
values.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf


class QuantumNATHybrid(tq.QuantumModule):
    """Variational quantum network that expects a 4‑D classical latent vector."""

    class QLayer(tq.QuantumModule):
        """Core variational layer with random and trainable gates."""

        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=2)
            self.crx0(qdev, wires=[0, 3])
            # Additional fixed gates for expressive power
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Encoder that maps classical values to a superposition
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Classical latent vector of shape (B, 4).

        Returns
        -------
        torch.Tensor
            Normalised quantum measurement vector of shape (B, 4).
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=True,
        )
        # Encode the classical data
        self.encoder(qdev, x)
        # Apply variational layer
        self.q_layer(qdev)
        # Measure
        out = self.measure(qdev)
        # Normalise
        return self.norm(out)


__all__ = ["QuantumNATHybrid"]
