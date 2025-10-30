"""Quantum module for the hybrid Natural Language Model (QML side)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QFCQuantum(tq.QuantumModule):
    """
    Two‑wire quantum module that encodes classical features, applies a
    parameterised random layer, and measures Pauli‑Z expectations.
    The module is fully differentiable and can be trained end‑to‑end
    with the classical CNN head.
    """

    class QLayer(tq.QuantumModule):
        """Parameterised quantum layer with random and rotation gates."""
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 2
            # Random layer with 20 operations for expressivity
            self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(self.n_wires)))
            # Learnable single‑qubit rotations
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            # Two‑qubit entangling gate
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=0)
            self.crx0(qdev, wires=[0, 1])
            tqf.hadamard(qdev, wires=1, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[0, 1], static=self.static_mode, parent_graph=self.graph)

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 2
        # Encoder that maps a 2‑dimensional vector to a quantum state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["2x2_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (batch, 2) containing classical features.

        Returns
        -------
        torch.Tensor
            Normalised expectation values of shape (batch, 2).
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=True,
        )
        # Encode classical data into the quantum state
        self.encoder(qdev, x)
        # Apply the parameterised quantum layer
        self.q_layer(qdev)
        # Measure Pauli‑Z on all qubits
        out = self.measure(qdev)
        # Normalise output for stability
        return self.norm(out)


__all__ = ["QFCQuantum"]
