import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple, Sequence

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

class FraudDetectionLayer(nn.Module):
    def __init__(self, params: FraudLayerParameters, clip: bool = True) -> None:
        super().__init__()
        weight = torch.tensor(
            [[params.bs_theta, params.bs_phi],
             [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32)
        bias = torch.tensor(params.phases, dtype=torch.float32)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)
        linear = nn.Linear(2, 2)
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)
        self.linear = linear
        self.activation = nn.Tanh()
        self.register_buffer("scale", torch.tensor(params.displacement_r, dtype=torch.float32))
        self.register_buffer("shift", torch.tensor(params.displacement_phi, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.activation(self.linear(x))
        return y * self.scale + self.shift

class _AutoencoderEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64)) -> None:
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class _AutoencoderDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int,
                 hidden_dims: Tuple[int, int] = (128, 64)) -> None:
        super().__init__()
        layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

class HybridFusionNet(nn.Module):
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64),
                 fraud_params: Sequence[FraudLayerParameters] | None = None,
                 output_dim: int = 1) -> None:
        super().__init__()
        self.encoder = _AutoencoderEncoder(input_dim, latent_dim, hidden_dims)
        self.decoder = _AutoencoderDecoder(latent_dim, input_dim, hidden_dims)
        self.fraud_layers = nn.ModuleList()
        if fraud_params:
            for p in fraud_params:
                self.fraud_layers.append(FraudDetectionLayer(p, clip=True))
        self.output_layer = nn.Linear(latent_dim, output_dim)
        self.quantum_latent = None

    def init_quantum(self, quantum_module) -> None:
        """Attach a quantum module that implements `run(thetas)` and returns a latent vector."""
        self.quantum_latent = quantum_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        if self.quantum_latent is not None:
            params = latent.detach().cpu().numpy()
            if params.ndim == 1:
                params = params.reshape(1, -1)
            q_latent = self.quantum_latent.run(params)
            q_latent = torch.as_tensor(q_latent, dtype=torch.float32, device=x.device)
            latent = latent + q_latent
        for layer in self.fraud_layers:
            latent = layer(latent)
        return self.output_layer(latent)

__all__ = ["HybridFusionNet", "FraudLayerParameters", "FraudDetectionLayer"]
