"""Hybrid classical quanvolution‑autoencoder for image classification.

This module extends the original ``QuanvolutionFilter`` by adding a lightweight
autoencoder that compresses the extracted patch features before classification.
The architecture keeps the vanilla convolutional filter for fast feature
extraction, then learns a latent representation with an MLP autoencoder,
and finally classifies the compressed vector with a linear head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionAutoEncoder(nn.Module):
    """Classical quanvolution filter + autoencoder + classifier.

    Parameters
    ----------
    in_channels : int
        Number of input channels (default 1 for MNIST).
    out_channels : int
        Number of output channels from the convolutional filter.
    kernel_size : int
        Size of the convolutional kernel.
    stride : int
        Stride of the convolution.
    latent_dim : int
        Dimensionality of the autoencoder latent space.
    hidden_dims : Tuple[int,...]
        Hidden layer sizes for the autoencoder encoder/decoder.
    dropout : float
        Dropout probability inside the autoencoder.
    num_classes : int
        Number of target classes for the final classifier.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        kernel_size: int = 2,
        stride: int = 2,
        latent_dim: int = 32,
        hidden_dims: tuple[int,...] = (128, 64),
        dropout: float = 0.1,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        # Classical quanvolution filter
        self.qfilter = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        # Compute feature vector size after flattening
        dummy = torch.zeros(1, in_channels, 28, 28)
        with torch.no_grad():
            feat = self.qfilter(dummy)
        feat_dim = feat.numel()
        # Autoencoder
        encoder_layers = []
        in_dim = feat_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, feat_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Classifier mapping latent space to logits
        self.classifier = nn.Linear(latent_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input through filter and autoencoder."""
        feat = self.qfilter(x).view(x.size(0), -1)
        latent = self.encoder(feat)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Reconstruct feature map from latent vector."""
        return self.decoder(latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class log‑softmax logits."""
        latent = self.encode(x)
        logits = self.classifier(latent)
        return F.log_softmax(logits, dim=-1)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Return reconstructed feature map for inspection."""
        latent = self.encode(x)
        return self.decode(latent)


__all__ = ["QuanvolutionAutoEncoder"]
