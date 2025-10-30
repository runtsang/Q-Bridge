import torch
from torch import nn
from typing import Iterable, Tuple

class FraudLayerParameters:
    """Parameters for a single fraud‑detection layer."""
    def __init__(self,
                 bs_theta: float,
                 bs_phi: float,
                 phases: Tuple[float, float],
                 squeeze_r: Tuple[float, float],
                 squeeze_phi: Tuple[float, float],
                 displacement_r: Tuple[float, float],
                 displacement_phi: Tuple[float, float],
                 kerr: Tuple[float, float]):
        self.bs_theta = bs_theta
        self.bs_phi = bs_phi
        self.phases = phases
        self.squeeze_r = squeeze_r
        self.squeeze_phi = squeeze_phi
        self.displacement_r = displacement_r
        self.displacement_phi = displacement_phi
        self.kerr = kerr

def _layer_from_params(params: FraudLayerParameters, clip: bool) -> nn.Module:
    weight = torch.tensor([[params.bs_theta, params.bs_phi],
                           [params.squeeze_r[0], params.squeeze_r[1]]],
                          dtype=torch.float32)
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
        def __init__(self):
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.activation(self.linear(x))
            return y * self.scale + self.shift
    return Layer()

def build_fraud_detection_model(input_params: FraudLayerParameters,
                                layers: Iterable[FraudLayerParameters]) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(l, clip=True) for l in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class AutoencoderConfig:
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64),
                 dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class AutoencoderNet(nn.Module):
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.encoder = self._build_mlp(config.input_dim,
                                       config.hidden_dims,
                                       config.latent_dim,
                                       config.dropout)
        self.decoder = self._build_mlp(config.latent_dim,
                                       tuple(reversed(config.hidden_dims)),
                                       config.input_dim,
                                       config.dropout)

    def _build_mlp(self,
                   in_dim: int,
                   hidden_dims: Tuple[int,...],
                   out_dim: int,
                   dropout: float) -> nn.Sequential:
        layers = []
        current = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(current, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current = h
        layers.append(nn.Linear(current, out_dim))
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def build_autoencoder(input_dim: int,
                      *,
                      latent_dim: int = 32,
                      hidden_dims: Tuple[int, int] = (128, 64),
                      dropout: float = 0.1) -> AutoencoderNet:
    cfg = AutoencoderConfig(input_dim,
                            latent_dim,
                            hidden_dims,
                            dropout)
    return AutoencoderNet(cfg)

class FraudDetectionAutoencoder:
    """Hybrid pipeline that compresses inputs with a classical autoencoder
    and classifies fraud risk with a photonic‑style network."""
    def __init__(self,
                 fraud_params: Iterable[FraudLayerParameters],
                 autoencoder_cfg: AutoencoderConfig):
        self.autoencoder = build_autoencoder(autoencoder_cfg.input_dim,
                                             latent_dim=autoencoder_cfg.latent_dim,
                                             hidden_dims=autoencoder_cfg.hidden_dims,
                                             dropout=autoencoder_cfg.dropout)
        self.fraud_model = build_fraud_detection_model(fraud_params[0],
                                                       list(fraud_params)[1:])
        self.projection = nn.Linear(autoencoder_cfg.latent_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.autoencoder.encode(x)
        z_proj = self.projection(z)
        return self.fraud_model(z_proj)
