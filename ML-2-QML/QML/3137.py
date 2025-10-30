"""Hybrid fraud‑detection model – quantum implementation.

This module implements :class:`FraudDetectionHybridModel` as a
quantum device that receives the 64‑dimensional feature vector from the
classical backbone, encodes it into a 4‑qubit register, applies a
parameterised circuit mirroring the photonic gate sequence, and
measures the qubits in the Pauli‑Z basis.
"""

from __future__ import annotations

import torch
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Optional


class FraudDetectionHybridModel(tq.QuantumModule):
    """
    Quantum submodule for the hybrid fraud‑detection pipeline.

    Parameters
    ----------
    n_qubits : int, default=4
        Number of qubits in the device.
    feature_dim : int, default=64
        Dimensionality of the incoming feature vector.
    """

    class QLayer(tq.QuantumModule):
        """Core variational block inspired by Quantum‑NAT."""

        def __init__(self, n_qubits: int = 4):
            super().__init__()
            self.n_qubits = n_qubits
            self.random_layer = tq.RandomLayer(
                n_ops=30, wires=list(range(self.n_qubits))
            )
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self, n_qubits: int = 4, feature_dim: int = 64):
        super().__init__()
        self.n_qubits = n_qubits
        self.feature_dim = feature_dim
        self.q_layer = self.QLayer(n_qubits)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(batch, feature_dim)`` from the classical
            feature extractor.

        Returns
        -------
        torch.Tensor
            Measurement outcomes of shape ``(batch, n_qubits)``.
        """
        batch_size = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_qubits,
            bsz=batch_size,
            device=x.device,
            record_op=True,
        )

        # --- Feature encoding: first ``n_qubits`` values as RX angles
        for i in range(self.n_qubits):
            tq.RX(has_params=True, trainable=False)(qdev, wires=i, param=x[:, i])

        # --- Variational block
        self.q_layer(qdev)

        # --- Measurement
        return self.measure(qdev)


__all__ = ["FraudDetectionHybridModel"]
