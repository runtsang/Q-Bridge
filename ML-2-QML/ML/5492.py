"""Hybrid quantum–classical kernel method integrating RBF kernels, self‑attention,
auto‑encoding and fraud‑detection layers."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------------------------------------------------
# Auto‑encoder
# ----------------------------------------------------------------------
class AutoencoderConfig:
    input_dim: int
    latent_dim: int
    hidden_dims: tuple[int, int]
    dropout: float

    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class AutoencoderNet(nn.Module):
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

def Autoencoder(input_dim: int, *, latent_dim: int = 32,
                hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1) -> AutoencoderNet:
    cfg = AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout)
    return AutoencoderNet(cfg)

# ----------------------------------------------------------------------
# Classical self‑attention
# ----------------------------------------------------------------------
class ClassicalSelfAttention:
    def __init__(self, embed_dim: int) -> None:
        self.embed_dim = embed_dim

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        q = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1),
                            dtype=torch.float32)
        k = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1),
                            dtype=torch.float32)
        v = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(q @ k.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ v).numpy()

# ----------------------------------------------------------------------
# Fraud‑detection layer
# ----------------------------------------------------------------------
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

    def __init__(self, bs_theta: float, bs_phi: float,
                 phases: tuple[float, float], squeeze_r: tuple[float, float],
                 squeeze_phi: tuple[float, float], displacement_r: tuple[float, float],
                 displacement_phi: tuple[float, float], kerr: tuple[float, float]):
        self.bs_theta = bs_theta
        self.bs_phi = bs_phi
        self.phases = phases
        self.squeeze_r = squeeze_r
        self.squeeze_phi = squeeze_phi
        self.displacement_r = displacement_r
        self.displacement_phi = displacement_phi
        self.kerr = kerr

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

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
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inp: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.activation(self.linear(inp))
            out = out * self.scale + self.shift
            return out

    return Layer()

def build_fraud_detection_program(input_params: FraudLayerParameters,
                                  layers: Iterable[FraudLayerParameters]) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(l, clip=True) for l in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# ----------------------------------------------------------------------
# Hybrid kernel method
# ----------------------------------------------------------------------
class HybridQuantumKernelMethod:
    """Classical hybrid kernel combining auto‑encoding, self‑attention and fraud detection."""

    def __init__(self, embed_dim: int = 4, latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64), gamma: float = 1.0,
                 fraud_input: FraudLayerParameters | None = None,
                 fraud_layers: Iterable[FraudLayerParameters] | None = None) -> None:
        self.autoencoder = Autoencoder(input_dim=embed_dim, latent_dim=latent_dim,
                                       hidden_dims=hidden_dims)
        self.attention = ClassicalSelfAttention(embed_dim)
        if fraud_input is None:
            fraud_input = FraudLayerParameters(0.0, 0.0, (0.0, 0.0),
                                               (0.0, 0.0), (0.0, 0.0),
                                               (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
        self.fraud_model = build_fraud_detection_program(fraud_input,
                                                          fraud_layers or [])
        self.gamma = gamma

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.autoencoder.encode(x)
        rotation = np.zeros((self.attention.embed_dim, self.attention.embed_dim))
        entangle = np.zeros((self.attention.embed_dim - 1,))
        attn = self.attention.run(rotation, entangle, latent.cpu().numpy())
        return torch.as_tensor(attn)

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_t = self._transform(a)
        b_t = self._transform(b)
        diff = a_t.unsqueeze(1) - b_t.unsqueeze(0)
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1))

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._transform(x)
        return self.fraud_model(feat)

__all__ = ["HybridQuantumKernelMethod", "Autoencoder", "AutoencoderNet",
           "ClassicalSelfAttention", "FraudLayerParameters",
           "build_fraud_detection_program"]
