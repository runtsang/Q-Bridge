import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple

class AutoencoderConfig:
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class AutoencoderNet(nn.Module):
    def __init__(self, config: AutoencoderConfig):
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

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))

def Autoencoder(input_dim: int, *, latent_dim: int = 32,
                hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1) -> AutoencoderNet:
    config = AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout)
    return AutoencoderNet(config)

class HybridConvNet(nn.Module):
    """Hybrid classical convolution + autoencoder + linear head."""
    def __init__(self, kernel_size: int = 2, stride: int = 1,
                 latent_dim: int = 32, hidden_dims: Tuple[int, int] = (128, 64),
                 dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=kernel_size, stride=stride, padding=0)
        dummy = torch.zeros(1, 1, 28, 28)
        out = self.conv(dummy)
        feat_dim = out.numel() // 1
        self.autoencoder = Autoencoder(input_dim=feat_dim,
                                       latent_dim=latent_dim,
                                       hidden_dims=hidden_dims,
                                       dropout=dropout)
        self.fc = nn.Linear(latent_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        flat = features.view(x.size(0), -1)
        latent = self.autoencoder.encode(flat)
        logits = self.fc(latent)
        return F.log_softmax(logits, dim=-1)

def Conv() -> HybridConvNet:
    """Return a hybrid convolutional filter suitable for MNIST."""
    return HybridConvNet()

__all__ = ["Conv"]
