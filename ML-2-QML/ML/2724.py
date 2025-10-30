from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple

# --------------------------------------------------------------------------- #
# Classical QCNN‑style feature extractor
# --------------------------------------------------------------------------- #
class QCNNFeatureExtractor(nn.Module):
    """
    Convolution‑inspired feature extractor mirroring the QCNNModel.
    It first projects the raw input into an 8‑dimensional space and then
    applies a small stack of fully‑connected layers with Tanh activations.
    """
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.pre = nn.Linear(input_dim, 8)
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return x


# --------------------------------------------------------------------------- #
# Configuration dataclass
# --------------------------------------------------------------------------- #
@dataclass
class HybridConfig:
    """
    Configuration for the hybrid autoencoder.
    """
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


# --------------------------------------------------------------------------- #
# Hybrid autoencoder: QCNN feature extractor + MLP encoder/decoder
# --------------------------------------------------------------------------- #
class HybridAutoencoderQCNN(nn.Module):
    """
    Hybrid autoencoder that first extracts features with a QCNN‑style
    convolutional block, then compresses them into a latent vector with an
    MLP encoder, and finally reconstructs the feature vector with a decoder.
    The output has the same dimensionality as the input feature extractor
    produces, not the raw input.
    """
    def __init__(self, config: HybridConfig) -> None:
        super().__init__()
        self.feature_extractor = QCNNFeatureExtractor(config.input_dim)

        # Determine size of the feature vector after extraction
        with torch.no_grad():
            dummy = torch.zeros(1, config.input_dim)
            feat_size = self.feature_extractor(dummy).shape[1]

        # Encoder
        encoder_layers = []
        in_dim = feat_size
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, feat_size))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature_extractor(x)
        z = self.encoder(feat)
        recon_feat = self.decoder(z)
        return recon_feat


# --------------------------------------------------------------------------- #
# Factory and training helper
# --------------------------------------------------------------------------- #
def HybridAutoencoderQCNNFactory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> HybridAutoencoderQCNN:
    """
    Factory that returns a configured HybridAutoencoderQCNN instance.
    """
    cfg = HybridConfig(input_dim, latent_dim, hidden_dims, dropout)
    return HybridAutoencoderQCNN(cfg)


def train_hybrid_autoencoder(
    model: HybridAutoencoderQCNN,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> list[float]:
    """
    Simple reconstruction training loop returning the loss history.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(torch.as_tensor(data, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = [
    "HybridAutoencoderQCNN",
    "HybridAutoencoderQCNNFactory",
    "train_hybrid_autoencoder",
    "HybridConfig",
]
