from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Iterable, Optional

# ---------- Autoencoder ----------
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        enc_layers = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                enc_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                dec_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def Autoencoder(input_dim: int, *, latent_dim: int = 32, hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1) -> AutoencoderNet:
    cfg = AutoencoderConfig(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims, dropout=dropout)
    return AutoencoderNet(cfg)

# ---------- Quanvolution ----------
class QuanvolutionFilter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        return feat.view(x.size(0), -1)

class QuanvolutionClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.fc = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.qfilter(x)
        logits = self.fc(feat)
        return F.log_softmax(logits, dim=-1)

# ---------- Estimator ----------
class EstimatorNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ---------- HybridQLSTM ----------
class HybridQLSTM(nn.Module):
    """
    Classical hybrid LSTM which optionally stacks an auto‑encoder,
    a quanvolution feature extractor, a recurrent core (nn.LSTM),
    and a regression head (EstimatorNN).  The class mimics the
    public API of the quantum version, enabling side‑by‑side
    experimentation.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        n_qubits: int = 0,          # ignored in classical branch
        use_autoencoder: bool = True,
        use_quanvolution: bool = True,
        use_estimator: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Optional modules
        self.autoencoder = Autoencoder(input_dim) if use_autoencoder else None
        self.quanvolution = QuanvolutionFilter() if use_quanvolution else None
        self.estimator = EstimatorNN() if use_estimator else None

        # Core recurrent layer
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expected shapes:
            * if use_autoencoder: (batch, seq_len, input_dim)
            * if use_quanvolution: (batch, seq_len, 1, 28, 28)
        """
        # Auto‑encoder stage
        if self.autoencoder is not None:
            b, t, d = x.shape
            flat = x.view(b * t, d)
            z = self.autoencoder.encode(flat)
            x = z.view(b, t, -1)

        # Quanvolution stage
        if self.quanvolution is not None:
            b, t, c, h, w = x.shape
            x = x.view(b * t, c, h, w)
            feat = self.quanvolution(x)
            x = feat.view(b, t, -1)

        # Recurrent core
        lstm_out, _ = self.lstm(x)

        # Estimator head
        if self.estimator is not None:
            lstm_out = self.estimator(lstm_out)

        return lstm_out

__all__ = [
    "HybridQLSTM",
    "AutoencoderNet",
    "Autoencoder",
    "QuanvolutionFilter",
    "QuanvolutionClassifier",
    "EstimatorNN",
]
