"""Hybrid autoencoder combining classical MLP, optional transformer encoder, and QCNN feature extractor.

The classical branch mirrors the original Autoencoder implementation but adds
an optional transformer encoder for sequential data and a QCNN feature
extractor for structured inputs.  The network can be used as a
stand‑alone PyTorch module or as a building block for larger models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional

import torch
from torch import nn
from torch.nn.functional import relu, F
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------
def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    tensor = data if isinstance(data, torch.Tensor) else torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
@dataclass
class HybridAutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    use_transformer: bool = False
    transformer_heads: int = 4
    transformer_ffn: int = 256
    transformer_blocks: int = 2
    use_qcnn: bool = False
    qcnn_depth: int = 3

# ----------------------------------------------------------------------
# Transformer helpers
# ----------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding used by the optional transformer."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class TransformerEncoderBlock(nn.Module):
    """A single transformer encoder block."""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.self_attn(src, src, src)
        src = self.norm1(src + self.dropout(attn_output))
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(src))))
        return self.norm2(src + self.dropout(ff_output))

class TransformerEncoder(nn.Module):
    """Stack of transformer encoder blocks."""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
             for _ in range(num_layers)]
        )
        self.pos_encoder = PositionalEncoding(d_model)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.pos_encoder(src)
        for layer in self.layers:
            src = layer(src)
        return src

# ----------------------------------------------------------------------
# QCNN feature extractor
# ----------------------------------------------------------------------
class QCNNFeatureMap(nn.Module):
    """A lightweight QCNN that transforms the input into a latent vector."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16), nn.Tanh(),
            nn.Linear(16, 32), nn.Tanh(),
            nn.Linear(32, output_dim), nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ----------------------------------------------------------------------
# Hybrid autoencoder
# ----------------------------------------------------------------------
class HybridAutoencoderNet(nn.Module):
    """Hybrid autoencoder that can optionally use a transformer encoder
    and/or a QCNN feature extractor before the classical MLP encoder."""
    def __init__(self, cfg: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Feature extraction stage
        if cfg.use_qcnn:
            self.feature_extractor = QCNNFeatureMap(cfg.input_dim, 32)
            feat_dim = 32
        else:
            self.feature_extractor = nn.Identity()
            feat_dim = cfg.input_dim

        # Optional transformer encoder
        if cfg.use_transformer:
            self.transformer = TransformerEncoder(
                d_model=feat_dim,
                nhead=cfg.transformer_heads,
                dim_feedforward=cfg.transformer_ffn,
                num_layers=cfg.transformer_blocks,
                dropout=cfg.dropout
            )
            enc_dim = feat_dim
        else:
            self.transformer = nn.Identity()
            enc_dim = feat_dim

        # Encoder MLP
        encoder_layers: List[nn.Module] = []
        in_dim = enc_dim
        for hidden in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder MLP
        decoder_layers: List[nn.Module] = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.transformer(x)
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------
def HybridAutoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
    use_transformer: bool = False,
    transformer_heads: int = 4,
    transformer_ffn: int = 256,
    transformer_blocks: int = 2,
    use_qcnn: bool = False,
    qcnn_depth: int = 3,
) -> HybridAutoencoderNet:
    cfg = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_transformer=use_transformer,
        transformer_heads=transformer_heads,
        transformer_ffn=transformer_ffn,
        transformer_blocks=transformer_blocks,
        use_qcnn=use_qcnn,
        qcnn_depth=qcnn_depth,
    )
    return HybridAutoencoderNet(cfg)

# ----------------------------------------------------------------------
# Training helper
# ----------------------------------------------------------------------
def train_hybrid_autoencoder(
    model: HybridAutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Training loop that mirrors the original `train_autoencoder`."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = [
    "HybridAutoencoder",
    "HybridAutoencoderNet",
    "HybridAutoencoderConfig",
    "train_hybrid_autoencoder",
]
