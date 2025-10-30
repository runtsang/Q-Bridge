import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuanvolutionFilter(nn.Module):
    """Classical 2×2 convolution that downsamples the input."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)

class SamplerModule(nn.Module):
    """Simple feed‑forward sampler that outputs a probability distribution."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)

class AutoencoderNet(nn.Module):
    """Fully‑connected autoencoder with configurable depth."""
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = input_dim
        for hidden in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))

class SelfAttentionModule(nn.Module):
    """Learnable self‑attention block that mimics the classical implementation."""
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.rotation = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle = nn.Parameter(torch.randn(embed_dim))
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        query = inputs @ self.rotation
        key = inputs @ self.entangle.unsqueeze(1)
        value = inputs
        scores = F.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return scores @ value

class HybridQuanvolutionNet(nn.Module):
    """Hybrid model that chains quanvolution, auto‑encoding, attention and sampling."""
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.autoencoder = AutoencoderNet(input_dim=4 * 14 * 14)
        self.attention = SelfAttentionModule(embed_dim=32)
        self.sampler = SamplerModule()
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)                      # (B, 4*14*14)
        latent = self.autoencoder.encode(features)      # (B, 32)
        attn_out = self.attention(latent)              # (B, 32)
        probs = self.sampler(attn_out)                 # (B, 2)
        logits = self.classifier(probs)                # (B, num_classes)
        return logits

__all__ = ["HybridQuanvolutionNet"]
