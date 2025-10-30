"""Hybrid kernel implementation for quantum machine learning.

The class implements a quantum kernel based on TorchQuantum.  It optionally
applies a quanvolution filter and an autoencoder to reduce dimensionality
before encoding the data into a quantum device.  A SamplerQNN is included
as a placeholder for variational parameter generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict


# --------------------------------------------------------------------------- #
# Autoencoder utilities (mirrors the classical version)
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(tq.QuantumModule):
    """Fully‑connected autoencoder implemented with TorchQuantum layers."""

    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        # Encoder
        layers = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            layers.append(tq.Linear(in_dim, hidden))
            layers.append(tq.ReLU())
            if cfg.dropout > 0.0:
                layers.append(tq.Dropout(cfg.dropout))
            in_dim = hidden
        layers.append(tq.Linear(in_dim, cfg.latent_dim))
        self.encoder = tq.Sequential(*layers)

        # Decoder
        layers = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            layers.append(tq.Linear(in_dim, hidden))
            layers.append(tq.ReLU())
            if cfg.dropout > 0.0:
                layers.append(tq.Dropout(cfg.dropout))
            in_dim = hidden
        layers.append(tq.Linear(in_dim, cfg.input_dim))
        self.decoder = tq.Sequential(*layers)

    @tq.static_support
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decoder(self.encoder(x))


# --------------------------------------------------------------------------- #
# Quanvolution filter
# --------------------------------------------------------------------------- #
class QuanvolutionFilter(tq.QuantumModule):
    """2‑pixel patch encoder using a 4‑qubit quantum kernel."""

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    @tq.static_support
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        # reshape to image patches
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                patches.append(self.measure(qdev).view(bsz, 4))
        return torch.cat(patches, dim=1)


# --------------------------------------------------------------------------- #
# SamplerQNN placeholder
# --------------------------------------------------------------------------- #
class SamplerQNN(tq.QuantumModule):
    """Simple variational sampler that returns a softmax over a linear layer."""

    def __init__(self) -> None:
        super().__init__()
        self.net = tq.Sequential(
            tq.Linear(2, 4),
            tq.Tanh(),
            tq.Linear(4, 2),
        )

    @tq.static_support
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return tq.functional.softmax(self.net(x), dim=-1)


# --------------------------------------------------------------------------- #
# Hybrid kernel
# --------------------------------------------------------------------------- #
class HybridKernel(tq.QuantumModule):
    """Quantum kernel with optional quanvolution & autoencoder preprocessing."""

    def __init__(self,
                 n_wires: int = 4,
                 use_autoencoder: bool = False,
                 autoencoder_cfg: AutoencoderConfig | None = None,
                 use_quanvolution: bool = False) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

        self.use_autoencoder = use_autoencoder
        self.use_quanvolution = use_quanvolution

        if self.use_autoencoder:
            if autoencoder_cfg is None:
                raise ValueError("autoencoder_cfg required when use_autoencoder=True")
            self.autoencoder = AutoencoderNet(autoencoder_cfg)

        if self.use_quanvolution:
            self.quanvolution = QuanvolutionFilter()

        self.sampler = SamplerQNN()  # placeholder for variational sampling

    @tq.static_support
    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_quanvolution:
            x = self.quanvolution(x)
        if self.use_autoencoder:
            x = self.autoencoder(x)
        return x

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        x = self._preprocess(x)
        y = self._preprocess(y)

        self.q_device.reset_states(x.shape[0])
        # encode x
        self.encoder(self.q_device, x)
        self.q_layer(self.q_device)
        # encode -y (reverse)
        self.encoder(self.q_device, -y)
        self.q_layer(self.q_device)

        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix between two datasets."""
        mat = [[self.forward(x, y).item() for y in b] for x in a]
        return np.array(mat)


__all__ = ["HybridKernel", "AutoencoderConfig", "AutoencoderNet", "QuanvolutionFilter", "SamplerQNN"]
