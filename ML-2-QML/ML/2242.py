import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Iterable, Tuple, List, Sequence

# --------------------------------------------------------------------------- #
# Classical encoder/decoder (auto‑encoder) ----------------------------------- #
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    """Configuration for the MLP autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    """A lightweight MLP autoencoder with configurable depth."""
    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        # Encoder
        encoder_layers: List[nn.Module] = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers: List[nn.Module] = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    """Factory that returns a configured AutoencoderNet."""
    cfg = AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout)
    return AutoencoderNet(cfg)


# --------------------------------------------------------------------------- #
# Photonic fraud‑detection core --------------------------------------------- #
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters for a single photonic layer."""
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
    """Builds a single photonic‑style layer as a PyTorch Module."""
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
            y = self.activation(self.linear(x))
            return y * self.scale + self.shift

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Builds a PyTorch sequential model that mimics the photonic circuit."""
    modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(l, clip=True) for l in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# --------------------------------------------------------------------------- #
# Hybrid fusion layer -------------------------------------------------------- #
# --------------------------------------------------------------------------- #
class FraudAutoEncoderHybrid(nn.Module):
    """
    Hybrid model that fuses a classical autoencoder with a photonic fraud‑detection
    backbone.  The autoencoder learns a compact latent representation of the
    transaction vector; this latent vector is projected to 2 D, fed through the
    photonic circuit, and the two representations are concatenated for final
    binary classification.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
    ) -> None:
        super().__init__()
        self.autoencoder = Autoencoder(
            input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
        )
        # Linear mapping from autoencoder latent to the 2‑D input of the circuit
        self.latent_to_2d = nn.Linear(latent_dim, 2)
        # Photonic fraud‑detection backbone (fixed parameters for demonstration)
        dummy_params = FraudLayerParameters(
            bs_theta=0.0,
            bs_phi=0.0,
            phases=(0.0, 0.0),
            squeeze_r=(0.0, 0.0),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(0.0, 0.0),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0),
        )
        self.fraud_circuit = build_fraud_detection_program(dummy_params, [])
        # Final classifier
        self.classifier = nn.Linear(latent_dim + 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical auto‑encoding
        latent = self.autoencoder.encode(x)
        # Map to 2‑D and run through photonic circuit
        q_input = self.latent_to_2d(latent)
        q_latent = self.fraud_circuit(q_input)
        # Fuse and classify
        combined = torch.cat([latent, q_latent], dim=-1)
        logits = self.classifier(combined)
        return torch.sigmoid(logits).squeeze(-1)


__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderNet",
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudAutoEncoderHybrid",
]
