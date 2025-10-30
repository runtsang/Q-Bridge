from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """
    Parameters that describe a single photonic‑inspired layer.
    The same dataclass is reused in the quantum implementation to keep the
    interface consistent across both domains.
    """
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]
    # Optional fields for random initialization
    init_weight_scale: float = 1.0
    init_bias: float = 0.0


def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to the range [-bound, bound]."""
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, clip: bool = False) -> nn.Module:
    """
    Build a single photonic‑inspired layer as a PyTorch module.
    The layer consists of a 2×2 linear transform, a tanh activation, and a
    channel‑wise affine rescaling that mimics displacement operations.
    """
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
            out = self.activation(self.linear(inputs))
            return out * self.scale + self.shift

    return Layer()


class FraudDetectionHybrid(nn.Module):
    """
    Classical fraud‑detection model that emulates the structure of a photonic circuit.
    The network consists of a stack of photonic‑inspired layers followed by a linear
    head that outputs a fraud probability.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        hidden_params: Sequence[FraudLayerParameters],
        final_bias: float = 0.0,
    ) -> None:
        super().__init__()
        modules = [_layer_from_params(input_params, clip=False)]
        modules.extend(_layer_from_params(p, clip=True) for p in hidden_params)
        modules.append(nn.Linear(2, 1))
        self.net = nn.Sequential(*modules)
        self.final_bias = nn.Parameter(torch.tensor(final_bias, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the photonic‑inspired network.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (batch, 2).

        Returns
        -------
        torch.Tensor
            Fraud score of shape (batch,).
        """
        out = self.net(x)
        return (out + self.final_bias).squeeze(-1)

    @classmethod
    def from_random(
        cls,
        num_hidden: int,
        seed: int | None = None,
        final_bias: float = 0.0,
    ) -> "FraudDetectionHybrid":
        """
        Create a network with randomly initialized photonic parameters.
        This mimics the stochastic nature of optical experiments and provides
        a baseline for comparison with the quantum variant.
        """
        rng = torch.Generator(device="cpu")
        if seed is not None:
            rng.manual_seed(seed)

        def rand_param() -> FraudLayerParameters:
            return FraudLayerParameters(
                bs_theta=torch.randn((), generator=rng).item(),
                bs_phi=torch.randn((), generator=rng).item(),
                phases=tuple(torch.randn(2, generator=rng).tolist()),
                squeeze_r=tuple(torch.randn(2, generator=rng).tolist()),
                squeeze_phi=tuple(torch.randn(2, generator=rng).tolist()),
                displacement_r=tuple(torch.randn(2, generator=rng).tolist()),
                displacement_phi=tuple(torch.randn(2, generator=rng).tolist()),
                kerr=tuple(torch.randn(2, generator=rng).tolist()),
                init_weight_scale=1.0,
                init_bias=0.0,
            )

        input_params = rand_param()
        hidden_params = [rand_param() for _ in range(num_hidden)]
        return cls(input_params, hidden_params, final_bias=final_bias)


__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
