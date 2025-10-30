"""Quantum‑enhanced quanvolution filter with variational circuits."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torch import Tensor
from dataclasses import dataclass
from typing import Optional

@dataclass
class QuantumConfig:
    """Runtime configuration for the variational circuit."""
    depth: int = 2
    n_ops: int = 8
    wires: list[int] | None = None


class QuanvolutionFilter(tq.QuantumModule):
    """Quantum filter that applies a learnable variational circuit to each 2×2 patch.

    The circuit depth is tunable (default 2), and the number of random
    layers is controlled by ``n_ops``.  The module uses a state‑vector
    simulator for exact measurement.

    The output is a 4‑dimensional feature vector per patch, which
    is concatenated into a single flat vector.
    """
    def __init__(self, cfg: QuantumConfig = QuantumConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.var_layer = tq.VariationalLayer(
            n_ops=self.cfg.n_ops,
            wires=list(range(self.n_wires)),
            depth=self.cfg.depth,
            optimizer=None,
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: Tensor) -> Tensor:
        bsz, _, H, W = x.shape
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []

        for r in range(0, H, 2):
            for c in range(0, W, 2):
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, patch)
                self.var_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))

        return torch.cat(patches, dim=1)


class QuanvolutionClassifier(nn.Module):
    """Hybrid network using the quantum filter followed by a small classical head."""
    def __init__(self, num_classes: int = 10, cfg: QuantumConfig = QuantumConfig()) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter(cfg)
        self.head = nn.Sequential(
            nn.Linear(4 * 14 * 14, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        features = self.qfilter(x)
        hidden = self.head(features)
        logits = self.classifier(hidden)
        return F.log_softmax(logits, dim=-1)

    def pretrain_step(self, x: Tensor, loss_fn: Optional[Tensor] = None) -> Tensor:
        if loss_fn is None:
            temp = 0.5
            aug1 = torch.flip(x, dims=[-1])
            aug2 = torch.roll(x, shifts=2, dims=[-1])
            z1 = self.qfilter(aug1)
            z2 = self.qfilter(aug2)
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)
            logits = torch.mm(z1, z2.t()) / temp
            N = logits.size(0)
            mask = torch.eye(N, dtype=torch.bool, device=logits.device)
            logits_mask = torch.ones_like(mask, dtype=torch.bool) & ~mask
            exp_logits = torch.exp(logits) * logits_mask
            denom = exp_logits.sum(dim=1, keepdim=True)
            pos_logits = torch.diag(logits)
            loss = -torch.log((torch.exp(pos_logits) + 1e-12) / (denom.squeeze() + 1e-12))
            return loss.mean()
        else:
            return loss_fn(self.qfilter(x), self.qfilter(x))

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier", "QuantumConfig"]
