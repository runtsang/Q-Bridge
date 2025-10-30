from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Tuple

# ---------- Dataset and Regression utilities ----------
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic data for regression or classification tasks.
    Inspired by the quantum regression example.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset wrapping the synthetic superposition data."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# ---------- Fraud‑Detection inspired custom layers ----------
@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

def build_custom_cnn(input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]) -> nn.Sequential:
    """
    Build a sequential model that mimics the photonic fraud‑detection program.
    The first layer is un‑clipped; subsequent layers are clipped.
    """
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# ---------- Autoencoder ----------
class AutoencoderNet(nn.Module):
    """
    Lightweight MLP autoencoder used as a feature regulariser.
    Mirrors the quantum autoencoder helper from the reference.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Encoder
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

        # Decoder
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

# ---------- Classical Hybrid Binary Classifier ----------
class HybridBinaryClassifier(nn.Module):
    """
    CNN + Autoencoder + Dense head classifier that mirrors the hybrid quantum model.
    The class is intentionally API‑compatible with the quantum variant.
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        autoencoder_input_dim: int = 55815,
        autoencoder_latent_dim: int = 32,
        autoencoder_hidden_dims: Tuple[int, int] = (128, 64),
        autoencoder_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Feature extractor
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully connected block
        self.fc1 = nn.Linear(autoencoder_input_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Autoencoder regulariser
        self.autoencoder = AutoencoderNet(
            input_dim=autoencoder_input_dim,
            latent_dim=autoencoder_latent_dim,
            hidden_dims=autoencoder_hidden_dims,
            dropout=autoencoder_dropout,
        )

        # Final classification head
        self.head = nn.Linear(1, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # --- CNN feature extraction ---
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)

        # --- Fully connected layers ---
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # --- Autoencoder regularisation (optional) ---
        latent = self.autoencoder.encode(x)
        recon = self.autoencoder.decode(latent)
        self.reg_loss = F.mse_loss(recon, x)

        # --- Final head ---
        logits = self.head(x)
        probs = F.softmax(logits, dim=-1)
        return probs

__all__ = [
    "HybridBinaryClassifier",
    "RegressionDataset",
    "generate_superposition_data",
    "build_custom_cnn",
    "FraudLayerParameters",
]
