import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

__all__ = [
    "Kernel",
    "KernelMatrix",
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "AutoencoderConfig",
    "AutoencoderNet",
    "Autoencoder",
    "SamplerQNN",
    "QuantumKernelAutoencoderFraudSampler",
]

# ------------------------------------------------------------------
# Hybrid RBF kernel with optional quantum augmentation
# ------------------------------------------------------------------
class Kernel(nn.Module):
    """Hybrid RBF kernel that can optionally incorporate a quantum kernel."""
    def __init__(self, gamma: float = 1.0, quantum_kernel_fn=None):
        super().__init__()
        self.gamma = gamma
        self.quantum_kernel_fn = quantum_kernel_fn

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        rbf = torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))
        if self.quantum_kernel_fn is not None:
            qk = self.quantum_kernel_fn(x, y)
            return rbf * qk
        return rbf

def KernelMatrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], kernel: Kernel) -> np.ndarray:
    """Compute Gram matrix using the provided kernel."""
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# ------------------------------------------------------------------
# Fraud detection style layer and network
# ------------------------------------------------------------------
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

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# ------------------------------------------------------------------
# Autoencoder
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# Classical SamplerQNN
# ------------------------------------------------------------------
class SamplerQNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.softmax(self.net(inputs), dim=-1)

# ------------------------------------------------------------------
# Combined wrapper
# ------------------------------------------------------------------
class QuantumKernelAutoencoderFraudSampler(nn.Module):
    """Encapsulates kernel, autoencoder, fraud net, and sampler."""
    def __init__(
        self,
        kernel: Kernel,
        autoencoder: AutoencoderNet,
        fraud_net: nn.Sequential,
        sampler: SamplerQNN,
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.autoencoder = autoencoder
        self.fraud_net = fraud_net
        self.sampler = sampler

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.autoencoder.encode(x)
        sampled = self.sampler(latent)
        fraud_logits = self.fraud_net(sampled)
        return fraud_logits
