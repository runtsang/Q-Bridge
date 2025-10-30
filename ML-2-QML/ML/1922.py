from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass
class FraudLayerParameters:
    """Parameters describing a fully‑connected layer in the classical model."""
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
    """Create a deterministic layer that mimics the photonic motif."""
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
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()


class BayesianLinear(nn.Module):
    """
    A Bayesian linear layer that samples weights and biases from a Gaussian
    posterior with trainable mean and log‑variance. The KL divergence with a
    standard normal prior is used as a regulariser.
    """
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features))
        self.weight_logvar = nn.Parameter(torch.randn(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.randn(out_features))
        self.bias_logvar = nn.Parameter(torch.randn(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps_w = torch.randn_like(self.weight_mu)
        eps_b = torch.randn_like(self.bias_mu)
        weight = self.weight_mu + torch.exp(0.5 * self.weight_logvar) * eps_w
        bias = self.bias_mu + torch.exp(0.5 * self.bias_logvar) * eps_b
        return nn.functional.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        weight_var = torch.exp(self.weight_logvar)
        bias_var = torch.exp(self.bias_logvar)
        kl_w = 0.5 * torch.sum(weight_var + self.weight_mu**2 - 1.0 - self.weight_logvar)
        kl_b = 0.5 * torch.sum(bias_var + self.bias_mu**2 - 1.0 - self.bias_logvar)
        return kl_w + kl_b


class FraudDetection(nn.Module):
    """
    Classical fraud‑detection model that combines a Bayesian prior layer with
    deterministic layers that emulate the photonic circuit.  The model is fully
    differentiable and exposes a `train_step` helper that performs a single
    optimisation step using a binary‑cross‑entropy loss plus a KL
    regulariser.
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> None:
        super().__init__()
        self.input_params = input_params
        self.layers_params = list(layers)
        self.bayes = BayesianLinear(2, 2)
        self.layers = nn.ModuleList(
            [_layer_from_params(p, clip=True) for p in [input_params] + self.layers_params]
        )
        self.out = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bayes(x)
        for layer in self.layers:
            out = layer(out)
        out = self.out(out)
        return torch.sigmoid(out)

    def train_step(
        self,
        optimizer: torch.optim.Optimizer,
        x: torch.Tensor,
        y: torch.Tensor,
        kl_weight: float = 1e-3,
    ) -> float:
        """
        Perform one optimisation step.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimiser that updates all parameters.
        x : Tensor
            Input features of shape (batch, 2).
        y : Tensor
            Binary labels of shape (batch, 1).
        kl_weight : float
            Weight of the KL divergence term.

        Returns
        -------
        loss.item() : float
            Scalar loss value after the step.
        """
        optimizer.zero_grad()
        logits = self.forward(x)
        loss = F.binary_cross_entropy(logits, y) + kl_weight * self.bayes.kl_divergence()
        loss.backward()
        optimizer.step()
        return loss.item()


__all__ = [
    "FraudLayerParameters",
    "FraudDetection",
    "BayesianLinear",
]
