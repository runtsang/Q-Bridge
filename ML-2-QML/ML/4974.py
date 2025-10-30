import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Tuple

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

class PhotonicLayer(nn.Module):
    """Classical analogue of a photonic layer used in the original photonic fraud detection circuit."""
    def __init__(self, params: FraudLayerParameters, clip: bool = True):
        super().__init__()
        weight = torch.tensor(
            [[params.bs_theta, params.bs_phi],
             [params.squeeze_r[0], params.squeeze_r[1]]],
            dtype=torch.float32
        )
        bias = torch.tensor(params.phases, dtype=torch.float32)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        self.activation = nn.Tanh()
        self.register_buffer("scale", torch.tensor(params.displacement_r, dtype=torch.float32))
        self.register_buffer("shift", torch.tensor(params.displacement_phi, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.linear(inputs))
        return x * self.scale + self.shift

class AutoencoderNet(nn.Module):
    """Light‑weight fully connected autoencoder used as feature extractor."""
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1):
        super().__init__()
        # Encoder
        encoder = []
        in_dim = input_dim
        for h in hidden_dims:
            encoder.append(nn.Linear(in_dim, h))
            encoder.append(nn.ReLU())
            if dropout > 0.0:
                encoder.append(nn.Dropout(dropout))
            in_dim = h
        encoder.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder)
        # Decoder
        decoder = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder.append(nn.Linear(in_dim, h))
            decoder.append(nn.ReLU())
            if dropout > 0.0:
                decoder.append(nn.Dropout(dropout))
            in_dim = h
        decoder.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

class QuanvolutionFilter(nn.Module):
    """2×2 convolutional filter inspired by the quanvolution example."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)

class FCL(nn.Module):
    """Simple fully‑connected layer that mimics the quantum FCL example."""
    def __init__(self, n_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(x)).mean(dim=0, keepdim=True)

class FraudDetectionHybrid(nn.Module):
    """
    Hybrid fraud‑detection model that fuses photonic layers, an autoencoder,
    a quanvolution filter, and a classical fully‑connected classifier.
    """
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64),
                 dropout: float = 0.1,
                 photonic_params: Iterable[FraudLayerParameters] | None = None):
        super().__init__()
        self.filter = QuanvolutionFilter()
        feature_dim = 4 * 14 * 14  # output of the filter
        self.autoencoder = AutoencoderNet(feature_dim, latent_dim, hidden_dims, dropout)
        self.photonic_layers = nn.ModuleList()
        if photonic_params:
            for i, params in enumerate(photonic_params):
                clip = i == 0
                self.photonic_layers.append(PhotonicLayer(params, clip=clip))
        self.classifier = FCL(n_features=latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x shape (batch, 1, 28, 28)
        qfeat = self.filter(x)
        latent = self.autoencoder(qfeat)
        for layer in self.photonic_layers:
            latent = layer(latent)
        out = self.classifier(latent)
        return out

__all__ = ["FraudDetectionHybrid", "FraudLayerParameters", "PhotonicLayer",
           "AutoencoderNet", "QuanvolutionFilter", "FCL"]
