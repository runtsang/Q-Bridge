"""Combined classical‑quantum estimator with autoencoder pre‑processing.

The model is a hybrid of:
* a lightweight autoencoder that compresses the raw input to a low‑dimensional latent space
* a differentiable quantum module (torchquantum) that learns a non‑linear mapping on the latent vector
* a final linear head that produces a scalar regression target

This design allows the classical encoder to remove redundancy while the quantum part explores
rich feature spaces that are hard to capture with vanilla neural nets.
"""

from __future__ import annotations

import torch
from torch import nn
import torchquantum as tq
from dataclasses import dataclass
from typing import Tuple

# ---------- Autoencoder ----------
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

# ---------- Quantum Module ----------
class QRegressor(tq.QuantumModule):
    """Differentiable quantum circuit that acts on a latent vector."""
    class _QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            # Random layer to provide expressivity
            self.random = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)

    def __init__(self, num_latent: int):
        super().__init__()
        self.n_wires = num_latent
        # Encoder that maps classical latent vector to a quantum state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_latent}xRy"])
        self.layer = self._QLayer(num_latent)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        bsz = latent.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=latent.device)
        # Encode classical data into qubits
        self.encoder(qdev, latent)
        # Apply variational layer
        self.layer(qdev)
        # Extract expectation values
        return self.measure(qdev)

# ---------- Hybrid Estimator ----------
class EstimatorQNNGen083(nn.Module):
    """
    Hybrid classical‑quantum estimator.

    Architecture:
        input  -> Autoencoder  -> latent
                 |              |
                 |              v
                 |          QuantumModule
                 |              |
                 |              v
                 |          Linear head -> scalar output
    """
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64),
                 dropout: float = 0.1) -> None:
        super().__init__()
        ae_cfg = AutoencoderConfig(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.autoencoder = AutoencoderNet(ae_cfg)
        self.quantum = QRegressor(num_latent=latent_dim)
        self.head = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        latent = self.autoencoder.encode(x)
        qfeat = self.quantum(latent)
        out = self.head(qfeat)
        return out.squeeze(-1)

__all__ = ["EstimatorQNNGen083", "AutoencoderNet", "AutoencoderConfig", "QRegressor"]
