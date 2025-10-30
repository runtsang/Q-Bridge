"""
Classical Fraud Detection model with Bayesian inference and probabilistic output.
"""

import dataclasses
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal


@dataclasses.dataclass
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


class BayesianLinear(nn.Module):
    """
    Bayesian linear layer with learnable mean and log‑variance for weights and biases.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_logvar = nn.Parameter(torch.full((out_features, in_features), -5.0))
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_logvar = nn.Parameter(torch.full((out_features,), -5.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_std = torch.exp(0.5 * self.weight_logvar)
        bias_std = torch.exp(0.5 * self.bias_logvar)
        weight = self.weight_mu + weight_std * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + bias_std * torch.randn_like(self.bias_mu)
        return F.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        weight_var = torch.exp(self.weight_logvar)
        bias_var = torch.exp(self.bias_logvar)
        weight_kl = 0.5 * (
            torch.sum(weight_var + self.weight_mu**2 - 1.0 - self.weight_logvar)
        )
        bias_kl = 0.5 * (
            torch.sum(bias_var + self.bias_mu**2 - 1.0 - self.bias_logvar)
        )
        return weight_kl + bias_kl


class FraudDetectionModel(nn.Module):
    """
    End‑to‑end fraud detection model that stacks classical layers, a Bayesian layer,
    and a probabilistic Bernoulli output.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: list[FraudLayerParameters],
    ) -> None:
        super().__init__()
        seq = [_layer_from_params(input_params, clip=False)]
        seq.extend(_layer_from_params(layer, clip=True) for layer in layers)
        seq.append(nn.Linear(2, 1))
        self.feature_extractor = nn.Sequential(*seq)
        self.bayesian = BayesianLinear(1, 1)
        self.classifier = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.bayesian(x)
        logits = x.squeeze(-1)
        probs = self.classifier(logits)
        return probs

    def kl_divergence(self) -> torch.Tensor:
        return self.bayesian.kl_divergence()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: list[FraudLayerParameters],
) -> nn.Sequential:
    """
    Convenience wrapper that returns the feature extractor part of the model.
    """
    seq = [_layer_from_params(input_params, clip=False)]
    seq.extend(_layer_from_params(layer, clip=True) for layer in layers)
    seq.append(nn.Linear(2, 1))
    return nn.Sequential(*seq)


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudDetectionModel",
    "BayesianLinear",
]
