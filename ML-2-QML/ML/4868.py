import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

# ----------------------------------------------------------------------
# Classical latent encoder/decoder
# ----------------------------------------------------------------------
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    """Encoder–decoder MLP that mirrors the quantum auto‑encoder of reference[2]."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            *self._make_blocks(cfg.input_dim, cfg.hidden_dims, cfg.dropout),
            nn.Linear(cfg.hidden_dims[-1], cfg.latent_dim),
        )
        self.decoder = nn.Sequential(
            *self._make_blocks(cfg.latent_dim, list(reversed(cfg.hidden_dims)), cfg.dropout),
            nn.Linear(cfg.hidden_dims[0], cfg.input_dim),
        )

    @staticmethod
    def _make_blocks(in_dim: int, dims: Sequence[int], dropout: float) -> Sequence[nn.Module]:
        blocks: list[nn.Module] = []
        for h in dims:
            blocks.append(nn.Linear(in_dim, h))
            blocks.append(nn.ReLU())
            if dropout > 0.0:
                blocks.append(nn.Dropout(dropout))
            in_dim = h
        return blocks

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))


# ----------------------------------------------------------------------
# Photonic layer parameters
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


# ----------------------------------------------------------------------
# Classical photonic‑style forward
# ----------------------------------------------------------------------
def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32
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

        def forward(self, inp: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.activation(self.linear(inp))
            return out * self.scale + self.shift

    return Layer()


# ----------------------------------------------------------------------
# Full hybrid model
# ----------------------------------------------------------------------
def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    auto_cfg: AutoencoderConfig,
) -> nn.Sequential:
    """Construct a hybrid model that stacks the photonic‐style layers with a
    latent auto‑encoder.  The encoder reduces dimensionality before the
    last linear output, and the decoder is discarded during inference."""
    # photonic layers
    modules: list[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(l, clip=True) for l in layers)

    # latent bottleneck
    ae = AutoencoderNet(auto_cfg)
    modules.append(ae.encode)          # only the encoder part

    # final classifier
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


__all__ = [
    "FraudLayerParameters",
    "AutoencoderConfig",
    "AutoencoderNet",
    "build_fraud_detection_program",
]
