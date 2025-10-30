from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QuanvolutionHybrid(tq.QuantumModule):
    """
    Quantum‑centric implementation of the hybrid quanvolution architecture.
    Combines a patch‑wise quantum filter with a variational QLayer
    that acts as a fully‑connected quantum head before a classical linear classifier.
    """

    class QLayer(tq.QuantumModule):
        """
        Variational layer inspired by the Quantum‑NAT QLayer.
        """
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

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.n_wires = 4
        # Encoder for 2×2 image patches (4 qubits)
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)
        self.classifier = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image batch of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (batch, num_classes).
        """
        bsz = x.shape[0]
        device = x.device

        # ----- Patch‑wise quantum filter -----
        qdev_filter = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                # Extract 2×2 patch and stack into (batch, 4)
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev_filter, data)
                self.q_layer(qdev_filter)
                measurement = self.measure(qdev_filter)
                patches.append(measurement.view(bsz, 4))
        patch_feats = torch.stack(patches, dim=1)          # (batch, 196, 4)
        agg_feats = patch_feats.mean(dim=1)               # (batch, 4)

        # ----- Quantum fully‑connected head -----
        qdev_fc = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        self.encoder(qdev_fc, agg_feats)
        self.q_layer(qdev_fc)
        out = self.measure(qdev_fc)
        out = self.norm(out)

        # ----- Classical linear classifier -----
        logits = self.classifier(out)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybrid"]
