"""
Hybrid Quanvolution model with graph aggregation and autoencoding.

This module keeps the public API from ``Quanvolution.py`` (``QuanvolutionFilter`` and
``QuanvolutionClassifier``) but enriches the feature extraction pipeline with:
  * 2×2 patch extraction via a 2‑D convolution,
  * a degree‑centrality graph computed from the patch embeddings,
  * a fully‑connected autoencoder that compresses the concatenated feature vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Iterable, List


# --------------------------------------------------------------------------- #
# Autoencoder utilities – identical to the original Autoencoder module
# --------------------------------------------------------------------------- #

@dataclass
class AutoencoderConfig:
    """Configuration for :class:`AutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    """Simple MLP autoencoder with configurable depth."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers: List[nn.Module] = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """Simple reconstruction training loop."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(data.to(device))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


# --------------------------------------------------------------------------- #
# Helper: degree‑centrality graph from patch embeddings
# --------------------------------------------------------------------------- #

def _degree_centrality(patches: torch.Tensor, threshold: float = 0.8) -> torch.Tensor:
    """
    Compute the degree centrality vector for each sample.

    Parameters
    ----------
    patches : torch.Tensor
        Tensor of shape (batch, num_patches, patch_dim) containing
        floating‑point patch embeddings.
    threshold : float
        Cosine‑similarity threshold used to create an undirected graph.

    Returns
    -------
    torch.Tensor
        Degree vector of shape (batch, num_patches).
    """
    # Normalise patches
    norm_patches = patches / (patches.norm(dim=-1, keepdim=True) + 1e-12)
    # Pairwise cosine similarity
    sims = torch.matmul(norm_patches, norm_patches.transpose(-2, -1))
    # Binary adjacency
    adj = (sims >= threshold).float()
    # Degree centrality
    return adj.sum(dim=-1)


# --------------------------------------------------------------------------- #
# Main quanvolution filter & classifier
# --------------------------------------------------------------------------- #

class QuanvolutionFilter(nn.Module):
    """
    Classical quanvolution filter with graph aggregation and autoencoding.

    The filter:
      1. Applies a 2×2 convolution to the input image.
      2. Computes degree‑centrality of the resulting patch embeddings.
      3. Concatenates the flattened convolution output with the degree vector.
      4. Passes the concatenated vector through a fully‑connected autoencoder.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        kernel_size: int = 2,
        stride: int = 2,
        hidden_dims: Tuple[int, int] = (128, 64),
        latent_dim: int = 32,
        dropout: float = 0.1,
        graph_threshold: float = 0.8,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        # Number of 2×2 patches in a 28×28 image
        self.num_patches = (28 // kernel_size) ** 2
        # Autoencoder receives concatenated conv output and graph degree vector
        input_dim = out_channels * self.num_patches + self.num_patches
        self.autoencoder = AutoencoderNet(
            AutoencoderConfig(
                input_dim=input_dim,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
            )
        )
        self.graph_threshold = graph_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass producing a latent representation.

        Parameters
        ----------
        x : torch.Tensor
            Input image of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Latent vector of shape (batch, latent_dim).
        """
        batch = x.shape[0]
        # Convolutional feature map
        conv_out = self.conv(x)  # (batch, out_channels, 14, 14)
        # Flatten to (batch, out_channels, num_patches)
        patches = conv_out.view(batch, conv_out.size(1), -1).permute(0, 2, 1)
        # Degree‑centrality graph vector
        degree = _degree_centrality(patches, threshold=self.graph_threshold)
        # Flatten conv output to (batch, out_channels * num_patches)
        flat_conv = conv_out.view(batch, -1)
        # Concatenate
        combined = torch.cat([flat_conv, degree], dim=1)
        # Autoencoder latent
        latent = self.autoencoder.encode(combined)
        return latent


class QuanvolutionClassifier(nn.Module):
    """
    Classifier that wraps :class:`QuanvolutionFilter` and a linear head.

    The classifier is compatible with the original API but now includes the
    graph‑aware, autoencoded latent representation.
    """
    def __init__(self, num_classes: int = 10, **filter_kwargs) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter(**filter_kwargs)
        self.linear = nn.Linear(self.qfilter.autoencoder.autoencoder[-1].out_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        latent = self.qfilter(x)
        logits = self.linear(latent)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
