"""Hybrid classical estimator that emulates quantum layers with classical surrogates.

The architecture mirrors the quantum example but replaces quantum primitives
with differentiable PyTorch modules, enabling fast training on CPU/GPU.
"""

import torch
from torch import nn
import numpy as np

# Classical surrogates for quantum components
class ConvFilter(nn.Module):
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # data shape: (batch, 1, H, W)
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[2, 3])  # collapse spatial dims

class ClassicalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.query_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_linear = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, rotation_params: torch.Tensor,
                entangle_params: torch.Tensor,
                inputs: torch.Tensor) -> torch.Tensor:
        query = self.query_linear(inputs)
        key = self.key_linear(inputs)
        scores = torch.softmax(torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)
        return torch.matmul(scores, inputs)

class FraudDetectionNet(nn.Module):
    def __init__(self,
                 input_params: dict,
                 layers: list[dict]) -> None:
        super().__init__()
        def layer_from_params(params: dict, clip: bool) -> nn.Module:
            weight = torch.tensor([[params['bs_theta'], params['bs_phi']],
                                   [params['squeeze_r'][0], params['squeeze_r'][1]]],
                                  dtype=torch.float32)
            bias = torch.tensor(params['phases'], dtype=torch.float32)
            if clip:
                weight = weight.clamp(-5.0, 5.0)
                bias = bias.clamp(-5.0, 5.0)
            linear = nn.Linear(2, 2)
            with torch.no_grad():
                linear.weight.copy_(weight)
                linear.bias.copy_(bias)
            activation = nn.Tanh()
            scale = torch.tensor(params['displacement_r'], dtype=torch.float32)
            shift = torch.tensor(params['displacement_phi'], dtype=torch.float32)

            class Layer(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.linear = linear
                    self.activation = activation
                    self.register_buffer('scale', scale)
                    self.register_buffer('shift', shift)

                def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                    out = self.activation(self.linear(inputs))
                    out = out * self.scale + self.shift
                    return out
            return Layer()

        modules = [layer_from_params(input_params, clip=False)]
        modules.extend(layer_from_params(l, clip=True) for l in layers)
        modules.append(nn.Linear(2, 1))
        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class EstimatorQNN(nn.Module):
    """Combined classical estimator that mimics the quantum workflow."""
    def __init__(self,
                 conv_kernel: int = 2,
                 attention_dim: int = 4,
                 fraud_params: dict | None = None,
                 fraud_layers: list[dict] | None = None) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size=conv_kernel)
        self.attention = ClassicalSelfAttention(embed_dim=attention_dim)
        if fraud_params is None:
            fraud_params = {'bs_theta': 0.0, 'bs_phi': 0.0, 'phases': (0.0, 0.0),
                           'squeeze_r': (0.0, 0.0),'squeeze_phi': (0.0, 0.0),
                            'displacement_r': (1.0, 1.0), 'displacement_phi': (0.0, 0.0)}
        if fraud_layers is None:
            fraud_layers = []
        self.fraud = FraudDetectionNet(fraud_params, fraud_layers)
        self.output_linear = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, H, W) image-like data
        conv_out = self.conv(x)
        # reshape to (batch, seq_len, embed_dim) for attention
        seq = conv_out.unsqueeze(-1)  # (batch, seq_len, 1)
        # rotation and entangle params as learnable parameters
        rotation_params = torch.randn(self.attention.embed_dim, requires_grad=True, device=x.device)
        entangle_params = torch.randn(self.attention.embed_dim, requires_grad=True, device=x.device)
        attn_out = self.attention(rotation_params, entangle_params, seq)
        fraud_in = attn_out.squeeze(-1)  # (batch, seq_len)
        fraud_out = self.fraud(fraud_in)
        return self.output_linear(fraud_out)

__all__ = ["EstimatorQNN"]
