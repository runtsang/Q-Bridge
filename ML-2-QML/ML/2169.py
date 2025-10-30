import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass
class FraudLayerParameters:
    """Parameters describing a single dense layer in the classical model."""
    weight: torch.Tensor  # 2×2 matrix
    bias: torch.Tensor    # 2‑dim vector
    scale: torch.Tensor   # 2‑dim vector
    shift: torch.Tensor   # 2‑dim vector
    activation: str = "tanh"

    def to_dict(self) -> dict:
        return {
            "weight": self.weight,
            "bias": self.bias,
            "scale": self.scale,
            "shift": self.shift,
            "activation": self.activation,
        }


def _layer_from_params(params: FraudLayerParameters, clip: bool = False) -> nn.Module:
    """Build a single dense block with optional clipping."""
    weight = params.weight.clone()
    bias = params.bias.clone()
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)

    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)

    act_cls = {"tanh": nn.Tanh, "relu": nn.ReLU, "sigmoid": nn.Sigmoid}[params.activation.lower()]
    activation = act_cls()

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", params.scale)
            self.register_buffer("shift", params.shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            h = self.activation(self.linear(inputs))
            return h * self.scale + self.shift

    return Layer()


class FraudDetectionModel(nn.Module):
    """Hybrid classical network for fraud detection with optional attention."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_layer_from_params(p, clip=(i > 0)) for i, p in enumerate([input_params] + list(layers))]
        )
        self.use_attention = use_attention
        if use_attention:
            # single‑head attention over the 2‑dim embeddings
            self.attn = nn.MultiheadAttention(embed_dim=2, num_heads=1, batch_first=True)
        self.out = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 2).

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch, 1).
        """
        out = x
        for layer in self.layers:
            out = layer(out)
        if self.use_attention:
            out, _ = self.attn(out.unsqueeze(0), out.unsqueeze(0), out.unsqueeze(0))
            out = out.squeeze(0)
        return self.out(out)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper that returns probabilities."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
