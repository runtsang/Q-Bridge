from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import torch
from torch import nn
import numpy as np

# ----------------------------------------------------------------------
# 1. Photonic‑style fraud layer parameters (classical analogue)
# ----------------------------------------------------------------------
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
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

# ----------------------------------------------------------------------
# 2. Autoencoder (fully‑connected, from reference pair 2)
# ----------------------------------------------------------------------
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

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

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(config)

# ----------------------------------------------------------------------
# 3. Quantum‑style fully‑connected layer (FCL) adapted to PyTorch
# ----------------------------------------------------------------------
class FCLayer(nn.Module):
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation

# ----------------------------------------------------------------------
# 4. Hybrid fraud detector
# ----------------------------------------------------------------------
class HybridFraudDetector(nn.Module):
    """
    Classical hybrid fraud detector that compresses raw transaction data with
    a fully‑connected autoencoder.  The latent representation is then fed
    through a stack of photonic‑style fraud layers followed by a quantum‑style
    fully‑connected layer (FCL).  The resulting model is a drop‑in replacement
    for the original quantum circuit while retaining the same logical flow.
    """
    def __init__(
        self,
        fraud_input_params: FraudLayerParameters,
        fraud_layers: Iterable[FraudLayerParameters],
        autoencoder_cfg: AutoencoderConfig,
    ) -> None:
        super().__init__()
        self.autoencoder = Autoencoder(
            input_dim=autoencoder_cfg.input_dim,
            latent_dim=autoencoder_cfg.latent_dim,
            hidden_dims=autoencoder_cfg.hidden_dims,
            dropout=autoencoder_cfg.dropout,
        )

        # Build fraud‑layer sequence
        modules = [_layer_from_params(fraud_input_params, clip=False)]
        modules.extend(_layer_from_params(layer, clip=True) for layer in fraud_layers)
        modules.append(nn.Linear(2, 1))
        self.classifier = nn.Sequential(*modules)

        # Quantum‑style FCL
        self.fcl = FCLayer(n_features=1)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode raw features into the latent space."""
        return self.autoencoder.encode(inputs)

    def classify(self, latent: torch.Tensor) -> torch.Tensor:
        """Run the fraud layers and FCL on the latent representation."""
        # Project latent to 2‑dimensional space expected by the fraud layers
        proj = nn.Linear(latent.size(-1), 2).to(latent.device)
        x = proj(latent)
        x = self.classifier(x)
        # FCL operates on a single feature; flatten the output
        out = self.fcl(x.squeeze(-1))
        return out

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = self.encode(inputs)
        return self.classify(latent)

__all__ = ["HybridFraudDetector", "FraudLayerParameters", "Autoencoder", "AutoencoderConfig"]
