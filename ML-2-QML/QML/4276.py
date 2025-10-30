"""Hybrid quantum‑classical network that fuses quanvolution, a quantum
fully‑connected layer, and a quantum autoencoder‑style compression.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchquantum as tq
import torchquantum.functional as tqf


class HybridQuanvolutionAutoencoder(tq.QuantumModule):
    """Quantum‑classical hybrid that mirrors the classical design in
    :class:`HybridQuanvolutionAutoencoder` but replaces the convolutional
    front‑end, fully‑connected projection, and autoencoder with
    equivalent quantum modules.
    """

    def __init__(self, num_classes: int = 10, latent_dim: int = 4) -> None:
        super().__init__()
        self.n_wires = 4  # number of qubits used in all sub‑circuits

        # Quantum filter (random layer)
        self.q_filter = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))

        # Quantum fully‑connected layer (parameterised by RX/RY/RZ/CRX)
        self.q_fc = self.QLayer()

        # Measurement producing a latent vector
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Classification head (classical linear layer)
        self.classifier = nn.Linear(latent_dim, num_classes)

    class QLayer(tq.QuantumModule):
        """Parameterised quantum layer inspired by the Quantum‑NAT QLayer."""

        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(
                n_ops=50, wires=list(range(self.n_wires))
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device

        # Flatten the input for quantum encoding
        x_flat = x.view(bsz, -1)

        # Quantum filter on the flattened input
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        for i in range(self.n_wires):
            theta = x_flat[:, i]
            tq.RY(has_params=False)(qdev, wires=i, params=theta)
        self.q_filter(qdev)

        # Quantum fully‑connected layer
        self.q_fc(qdev)

        # Quantum autoencoder‑style compression via a second random layer
        tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))(qdev)

        # Measurement producing latent vector
        latent = self.measure(qdev).view(bsz, -1)

        # Classical classification head
        logits = self.classifier(latent)
        return F.log_softmax(logits, dim=-1)


__all__ = ["HybridQuanvolutionAutoencoder"]
