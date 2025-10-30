from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Tuple

# Import the quantum sampler from the quantum module
# (the quantum implementation is defined in a separate file)
from.quantum_sampler import SamplerQNN as QuantumSampler


@dataclass
class SamplerParams:
    """Parameters for a single classical layer in the hybrid sampler."""
    theta: float
    phi: float
    scale: Tuple[float, float]
    shift: Tuple[float, float]


def _make_layer(params: SamplerParams, clip: bool = False) -> nn.Module:
    """Construct a 2→2 linear layer with tanh activation, optional clipping."""
    weight = torch.tensor(
        [[params.theta, params.phi],
         [params.scale[0], params.scale[1]]],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.shift, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    return nn.Sequential(linear, nn.Tanh())


class SamplerQNN(nn.Module):
    """
    Hybrid classical‑quantum sampler that mirrors the fraud‑detection style
    layer stack and mixes its output with a quantum sampler.
    """
    def __init__(
        self,
        classical_params: SamplerParams,
        classical_layers: Iterable[SamplerParams],
        quantum_mix: float = 0.5,
    ) -> None:
        super().__init__()
        # Classical backbone
        modules = [_make_layer(classical_params, clip=False)]
        modules.extend(_make_layer(l, clip=True) for l in classical_layers)
        modules.append(nn.Linear(2, 2))  # produce two logits
        self.classical = nn.Sequential(*modules)

        # Quantum sampler
        self.quantum = QuantumSampler()
        self.quantum_mix = quantum_mix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return a probability vector over 4 outcomes. The first two entries
        come from the classical sampler (logits → softmax), the last two
        come from the quantum sampler. The two distributions are mixed
        according to ``quantum_mix``.
        """
        # Classical logits
        logits = self.classical(x)
        probs_classical = F.softmax(logits, dim=-1)

        # Quantum probabilities
        probs_quantum = self.quantum(x)

        # Combine: keep first two from classical, last two from quantum
        combined = torch.cat([probs_classical, probs_quantum[..., :2]], dim=-1)

        if self.quantum_mix < 1.0:
            combined = self.quantum_mix * probs_quantum + (1 - self.quantum_mix) * combined

        return combined

    def sample(self, n_samples: int, device: str = "cpu") -> torch.Tensor:
        """
        Draw samples from the joint distribution.
        """
        rng = torch.randn(n_samples, 2, device=device)
        probs = self.forward(rng)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


__all__ = ["SamplerQNN", "SamplerParams"]
