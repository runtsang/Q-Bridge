from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple

# ----------------------------------------------------------------------
# Classical QCNN – a dropout‑enhanced, batch‑norm‑aware variant
# ----------------------------------------------------------------------
class QCNNModel(nn.Module):
    def __init__(self, in_features: int = 8, hidden_features: int = 16) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.Tanh()
        )
        self.conv1 = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.Tanh()
        )
        self.pool1 = nn.Sequential(
            nn.Linear(hidden_features, 12),
            nn.BatchNorm1d(12),
            nn.Tanh()
        )
        self.conv2 = nn.Sequential(
            nn.Linear(12, 8),
            nn.BatchNorm1d(8),
            nn.Tanh()
        )
        self.pool2 = nn.Sequential(
            nn.Linear(8, 4),
            nn.BatchNorm1d(4),
            nn.Tanh()
        )
        self.conv3 = nn.Sequential(
            nn.Linear(4, 4),
            nn.BatchNorm1d(4),
            nn.Tanh()
        )
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

# ----------------------------------------------------------------------
# Classical Autoencoder – configurable encoder/decoder
# ----------------------------------------------------------------------
class AutoencoderConfig:
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

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

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

# ----------------------------------------------------------------------
# Classical Sampler Network – lightweight softmax output
# ----------------------------------------------------------------------
class SamplerModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)

# ----------------------------------------------------------------------
# Helper to build a classical classifier
# ----------------------------------------------------------------------
def build_classifier_circuit(num_features: int, depth: int) -> nn.Sequential:
    layers = []
    in_dim = num_features
    for _ in range(depth):
        layers.append(nn.Linear(in_dim, num_features))
        layers.append(nn.ReLU())
        in_dim = num_features
    layers.append(nn.Linear(in_dim, 2))
    return nn.Sequential(*layers)

# ----------------------------------------------------------------------
# Hybrid model integrating all components
# ----------------------------------------------------------------------
class HybridQCNN(nn.Module):
    """
    A hybrid architecture that chains a convolution‑style network (QCNNModel),
    a variational autoencoder (AutoencoderNet), a feed‑forward classifier
    (built by build_classifier_circuit), and a lightweight sampler (SamplerModule).
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        num_features: int = 8,
        classifier_depth: int = 3
    ) -> None:
        super().__init__()
        self.qcnn = QCNNModel(in_features=input_dim)
        self.autoencoder = AutoencoderNet(
            AutoencoderConfig(input_dim, latent_dim, hidden_dims)
        )
        self.classifier = build_classifier_circuit(num_features, classifier_depth)
        self.sampler = SamplerModule()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.qcnn(x)
        x = self.autoencoder(x)
        x = self.classifier(x)
        return self.sampler(x)

__all__ = ["HybridQCNN", "QCNNModel", "AutoencoderNet", "SamplerModule", "build_classifier_circuit"]
