import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple

@dataclass
class AutoencoderConfig:
    """Configuration for the dense auto‑encoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class DenseAutoencoder(nn.Module):
    """Simple MLP auto‑encoder with configurable hidden layers."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dims[0], cfg.hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dims[1], cfg.latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dims[1], cfg.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dims[0], cfg.input_dim)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

class QFCEncoder(nn.Module):
    """CNN encoder inspired by QFCModel."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        f = f.view(f.size(0), -1)
        f = self.fc(f)
        return self.norm(f)

class SamplerDecoder(nn.Module):
    """Classical sampler network mirroring SamplerQNN."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)

class UnifiedAutoencoder(nn.Module):
    """
    Hybrid auto‑encoder that can operate in dense, CNN, or sequence modes.
    The same public API is shared with the quantum implementation.
    """
    def __init__(
        self,
        cfg: AutoencoderConfig,
        *,
        use_cnn: bool = False,
        use_lstm: bool = False,
        lstm_hidden_dim: int = 64,
        vocab_size: int = 1000,
        tagset_size: int = 10
    ) -> None:
        super().__init__()
        self.cfg = cfg
        if use_cnn:
            self.encoder = QFCEncoder()
            # map CNN output to latent dimension
            self.latent_proj = nn.Linear(4, cfg.latent_dim)
        else:
            self.encoder = DenseAutoencoder(cfg)
        if use_lstm:
            self.lstm = nn.LSTM(input_size=cfg.latent_dim, hidden_size=lstm_hidden_dim, batch_first=True)
            self.lstm_proj = nn.Linear(lstm_hidden_dim, cfg.latent_dim)
        else:
            self.lstm = None
        self.decoder = SamplerDecoder()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        if hasattr(z, "shape") and len(z.shape) == 2 and z.shape[1] == 4:
            z = self.latent_proj(z)
        if self.lstm is not None:
            z, _ = self.lstm(z.unsqueeze(1))
            z = z.squeeze(1)
            z = self.lstm_proj(z)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

__all__ = ["AutoencoderConfig", "DenseAutoencoder", "QFCEncoder", "SamplerDecoder", "UnifiedAutoencoder"]
