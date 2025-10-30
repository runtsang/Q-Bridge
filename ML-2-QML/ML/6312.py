"""
Hybrid fraud‑detection module that couples a classical autoencoder with a
photonic‑style variational classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
#  Classical autoencoder
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    """Configuration for the autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    """Simple fully‑connected autoencoder."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        # Encoder
        enc_layers: list[nn.Module] = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, hidden))
            enc_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                enc_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers: list[nn.Module] = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, hidden))
            dec_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                dec_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        dec_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))


# --------------------------------------------------------------------------- #
#  Photonic‑style variational classifier
# --------------------------------------------------------------------------- #
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


def _build_photonic_layer(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Create a single photonic layer as a PyTorch module."""
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

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.activation(self.linear(x))
            out = out * self.scale + self.shift
            return out

    return Layer()


def _build_photonic_sequence(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules: list[nn.Module] = [_build_photonic_layer(input_params, clip=False)]
    modules.extend(_build_photonic_layer(l, clip=True) for l in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# --------------------------------------------------------------------------- #
#  Hybrid model: autoencoder + photonic classifier
# --------------------------------------------------------------------------- #
class FraudDetectionAutoencoder(nn.Module):
    """
    A hybrid model that first encodes the input transaction features into a
    low‑dimensional latent vector using a classical autoencoder, then passes
    the latent vector through a photonic‑style variational classifier.
    """
    def __init__(
        self,
        ae_cfg: AutoencoderConfig,
        photonic_params: FraudLayerParameters,
        photonic_layers: Iterable[FraudLayerParameters],
    ) -> None:
        super().__init__()
        self.autoencoder = AutoencoderNet(ae_cfg)
        self.classifier = _build_photonic_sequence(photonic_params, photonic_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.autoencoder.encode(x)
        # Reduce latent dimension to 2 for the photonic classifier
        # (simple linear projection)
        z_proj = nn.functional.linear(z, torch.randn(2, z.size(-1), device=z.device))
        return self.classifier(z_proj)


# --------------------------------------------------------------------------- #
#  Training utilities
# --------------------------------------------------------------------------- #
def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


def train_fraud_detection(
    model: FraudDetectionAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Train the hybrid model.  Returns the reconstruction loss history."""
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
    "AutoencoderConfig",
    "AutoencoderNet",
    "FraudLayerParameters",
    "FraudDetectionAutoencoder",
    "train_fraud_detection",
]
