from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QLayer(tq.QuantumModule):
    """A parameterised quantum layer used inside the encoder."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
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


class QuantumDecoder(tq.QuantumModule):
    """Decoder that maps a 4‑dimensional latent vector back to a feature vector."""
    def __init__(self, n_wires: int, latent_dim: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.latent_dim = latent_dim
        # Encode latent into qubits via Ry gates
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(latent_dim)]
        )
        self.quantum_layer = tq.RandomLayer(n_ops=10, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(qdev)
        self.quantum_layer(qdev)
        return self.measure(qdev)


class HybridNATAutoEncoder(tq.QuantumModule):
    """
    Quantum‑centric counterpart of the classical HybridNATAutoEncoder.
    Implements a full quantum encoder–decoder pipeline while preserving
    the 4‑dimensional NAT output and adding a latent reconstruction branch.
    """
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4

        # Classical feature extractor (same as in the classical model)
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])

        # Quantum encoder block
        self.q_layer = QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

        # Quantum decoder
        self.decoder = QuantumDecoder(n_wires=self.n_wires, latent_dim=self.n_wires)
        self.decoder_norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            latent: 4‑dimensional quantum NAT output
            recon: 4‑dimensional reconstruction from the quantum decoder
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=True,
        )
        # Pooling to match the feature dimensionality used by the encoder
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        latent = self.measure(qdev)
        latent = self.norm(latent)

        # Decoder branch
        qdev_dec = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=x.device,
        )
        self.decoder.encoder(qdev_dec, latent)
        self.decoder.quantum_layer(qdev_dec)
        recon = self.decoder.measure(qdev_dec)
        recon = self.decoder_norm(recon)

        return latent, recon


__all__ = ["HybridNATAutoEncoder"]
