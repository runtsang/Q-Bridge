"""Enhanced quantum version of QFCModel.

The model now includes a parameter‑shared entangling layer and a helper
to compute a measurement‑based loss.  It remains a drop‑in replacement
for the seed model but can be run on a real device or a simulator.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QFCModel(tq.QuantumModule):
    """Quantum CNN‑to‑FC model with shared entangling layer and loss helper."""

    class QLayer(tq.QuantumModule):
        """Parameter‑shared entangling block.

        The same rotation parameters are applied to every wire,
        reducing the parameter count while still enabling entanglement
        via a global CRX gate.
        """

        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            # Shared rotation parameters
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            # Global entangler
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            # Apply the same rotation to every wire
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)
                self.rz(qdev, wires=w)
            # Entangle all pairs with a single CRX
            for w in range(self.n_wires - 1):
                self.crx(qdev, wires=[w, w + 1])

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        # Encoder: maps classical features to quantum state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer(self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 1, 28, 28).

        Returns:
            Normalised expectation values of shape (batch, 4).
        """
        bsz = x.shape[0]
        # Reduce spatial resolution to 16 features per sample
        pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, 16)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)  # shape (batch, n_wires)
        return self.norm(out)

    # ------------------------------------------------------------------
    # Helper methods for quantum‑specific loss evaluation
    # ------------------------------------------------------------------
    def expectation(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw expectation values without normalisation."""
        bsz = x.shape[0]
        pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, 16)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        return self.measure(qdev)

    @staticmethod
    def loss_fn(expectation: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean‑squared‑error loss between expectation and target."""
        return F.mse_loss(expectation, target)

    @staticmethod
    def predict(expectation: torch.Tensor) -> torch.Tensor:
        """Convert expectation values to class logits via a linear map."""
        # Simple linear mapping for demonstration
        linear = nn.Linear(4, 4, bias=False).to(expectation.device)
        return linear(expectation)

__all__ = ["QFCModel"]
