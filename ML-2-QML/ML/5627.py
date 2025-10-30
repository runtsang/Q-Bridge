import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable
import numpy as np

# --- Photonic fraud‑detection components (from reference 1) ---
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

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# --- Classical convolution filter (from reference 4) ---
class ConvFilter(nn.Module):
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        tensor = data.view(-1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=(1, 2, 3)).unsqueeze(-1)

# --- Classical self‑attention (from reference 3) ---
class ClassicalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.weight_query = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.weight_key = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        query = inputs @ self.weight_query
        key = inputs @ self.weight_key
        value = inputs
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return scores @ value

# --- RBF kernel (from reference 2) ---
class Kernel(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

# --- Hybrid model ---
class FraudDetectionHybrid(nn.Module):
    def __init__(
        self,
        conv_kernel: int = 2,
        attention_dim: int = 4,
        gamma: float = 1.0,
        fraud_params: FraudLayerParameters | None = None,
        layers_params: list[FraudLayerParameters] | None = None,
    ) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size=conv_kernel, threshold=0.0)
        self.attention = ClassicalSelfAttention(embed_dim=attention_dim)
        self.kernel = Kernel(gamma=gamma)
        self.fc = nn.Linear(attention_dim + 2, 2)
        if fraud_params is None:
            fraud_params = FraudLayerParameters(
                bs_theta=0.0,
                bs_phi=0.0,
                phases=(0.0, 0.0),
                squeeze_r=(0.0, 0.0),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(0.0, 0.0),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            )
        if layers_params is None:
            layers_params = []
        self.fraud_network = build_fraud_detection_program(fraud_params, layers_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolution feature
        conv_out = self.conv(x)  # (batch,1)
        # Self‑attention feature
        attn_in = x.view(x.size(0), -1)
        attn_in = attn_in[:, :self.attention.embed_dim]
        attn_out = self.attention(attn_in)  # (batch, attention_dim)
        # Kernel between convolution and attention
        kernel_val = self.kernel(conv_out, attn_out)  # (batch,1)
        # Concatenate features
        features = torch.cat([conv_out, attn_out, kernel_val], dim=-1)  # (batch, attention_dim+2)
        hidden = self.fc(features)  # (batch,2)
        return self.fraud_network(hidden)
