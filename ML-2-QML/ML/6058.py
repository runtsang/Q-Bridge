"""
Hybrid autoencoder with fraud‑detection inspired classifier.
Combines ideas from the classical autoencoder and fraud detection
seed projects into a single PyTorch module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
#  Fraud‑detection style parameterised layer
# --------------------------------------------------------------------------- #

@dataclass
class FraudLayerParameters:
    """Parameters for a single fraud‑detection layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
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
            x = self.activation(self.linear(inputs))
            return x * self.scale + self.shift

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential model that mirrors the photonic fraud detection circuit."""
    modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
#  Autoencoder architecture
# --------------------------------------------------------------------------- #

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """Lightweight fully‑connected autoencoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        # Encoder
        encoder_layers: List[nn.Module] = []
        in_dim = config.input_dim
        for h in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers: List[nn.Module] = []
        in_dim = config.latent_dim
        for h in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(cfg)

# --------------------------------------------------------------------------- #
#  Hybrid model: autoencoder + fraud‑detection classifier
# --------------------------------------------------------------------------- #

class HybridAutoencoderFraudDetector(nn.Module):
    """
    Combines an autoencoder with a fraud‑detection classifier.
    The classifier operates on the latent representation produced by the encoder.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        fraud_layers: Iterable[FraudLayerParameters] | None = None,
    ) -> None:
        super().__init__()
        self.autoencoder = Autoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        # Build fraud‑detection classifier on latent space
        if fraud_layers is None:
            fraud_layers = []
        # Use a dummy input layer to match 2‑dimensional fraud layers
        dummy_input = FraudLayerParameters(
            bs_theta=0.0, bs_phi=0.0,
            phases=(0.0, 0.0),
            squeeze_r=(0.0, 0.0),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(0.0, 0.0),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0),
        )
        self.classifier = build_fraud_detection_program(dummy_input, fraud_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(z)

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        """Return logits for fraud classification."""
        return self.classifier(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        logits = self.classify(z)
        return recon, logits

def train_hybrid(
    model: HybridAutoencoderFraudDetector,
    data: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> Tuple[List[float], List[float]]:
    """
    Train the hybrid model.
    Returns (reconstruction_loss_history, classification_loss_history).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data), _as_tensor(labels))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    rec_hist: List[float] = []
    cls_hist: List[float] = []

    for _ in range(epochs):
        epoch_rec = 0.0
        epoch_cls = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float()
            optimizer.zero_grad(set_to_none=True)
            recon, logits = model(batch_x)
            loss_rec = mse_loss(recon, batch_x)
            loss_cls = bce_loss(logits.squeeze(), batch_y)
            loss = loss_rec + loss_cls
            loss.backward()
            optimizer.step()
            epoch_rec += loss_rec.item() * batch_x.size(0)
            epoch_cls += loss_cls.item() * batch_x.size(0)
        rec_hist.append(epoch_rec / len(dataset))
        cls_hist.append(epoch_cls / len(dataset))

    return rec_hist, cls_hist

# --------------------------------------------------------------------------- #
#  Utility
# --------------------------------------------------------------------------- #

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "AutoencoderNet",
    "Autoencoder",
    "HybridAutoencoderFraudDetector",
    "train_hybrid",
]
