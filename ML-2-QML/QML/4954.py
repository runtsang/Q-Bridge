"""Quantum hybrid model with encoder, variational layer, and regression head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class HybridQuantumNAT(tq.QuantumModule):
    """
    Quantum analogue of the classical HybridQuantumNAT.
    Encodes a 2‑D feature map into a quantum state, processes it with a
    variational circuit, measures, and feeds the expectation values into
    a classical regression head.  The sampler and estimator sub‑modules
    are omitted in the quantum version to keep the circuit lightweight.
    """

    class QLayer(tq.QuantumModule):
        """Variational layer combining random gates and trainable rotations."""
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(
                n_ops=50, wires=list(range(n_wires))
            )
            # Trainable single‑qubit rotations
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)
                self.rz(qdev, wires=w)

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        # Encoder that maps a 2‑D feature vector into a quantum state
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{n_wires}xRy"]
        )
        self.q_layer = self.QLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(n_wires, 4)  # regression head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: batch of 2‑D feature vectors (shape: [N, D]).
        Returns:
            Regression output (shape: [N, 4]).
        """
        bsz = x.shape[0]
        # Reduce the image to a 1‑D feature vector (as in the classical version)
        pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, 16)
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=True,
        )
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


__all__ = ["HybridQuantumNAT"]
