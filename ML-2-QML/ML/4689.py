"""Hybrid sampler combining a classical convolutional filter with a variational sampler network.

The module exposes two factories:
  * :func:`Conv` – a lightweight convolutional filter that mimics a quanvolution layer.
  * :func:`SamplerQNN` – a neural network that uses the convolutional filter to provide a feature
    and a soft‑max sampler output.  The network can be wrapped in :class:`FastEstimator`
    (defined in :mod:`FastBaseEstimator`) to add Gaussian shot noise.

The design follows the structure of the original ``SamplerQNN.py`` but augments it
with the convolution and estimation utilities from the supplied references.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, List, Callable, Sequence

# --------------------------------------------------------------------------- #
# 1. Convolutional filter – emulates the quanvolution layer
# --------------------------------------------------------------------------- #
def Conv(kernel_size: int = 2,
         threshold: float = 0.0) -> nn.Module:
    """Return a PyTorch module that mimics a quantum filter."""
    class ConvFilter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kernel_size = kernel_size
            self.threshold = threshold
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        def forward(self, data: torch.Tensor) -> torch.Tensor:
            """Return the mean activation over the filter window."""
            # ``data`` is expected to be a 2‑D array matching ``kernel_size``.
            tensor = torch.as_tensor(data, dtype=torch.float32)
            tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
            logits = self.conv(tensor)
            activations = torch.sigmoid(logits - self.threshold)
            return activations.mean()

    return ConvFilter()


# --------------------------------------------------------------------------- #
# 2. Sampler network – soft‑max output + optional convolutional feature
# --------------------------------------------------------------------------- #
def SamplerQNN(use_conv: bool = False) -> nn.Module:
    """Return a neural network that can act as a classical sampler."""
    class SamplerModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
            )
            self.conv = Conv() if use_conv else None

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            """Return the soft‑max probabilities of the sampler."""
            logits = self.net(inputs)
            probs = F.softmax(logits, dim=-1)
            if self.conv is not None:
                # ``inputs`` are expected to be 2‑D images of size 2×2.
                conv_out = self.conv(inputs)
                # Concatenate the scalar feature to the probability vector.
                probs = torch.cat([probs, conv_out.unsqueeze(-1)], dim=-1)
            return probs

    return SamplerModule()


# --------------------------------------------------------------------------- #
# 3. Fast estimator – optional Gaussian noise
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """Evaluates a PyTorch model on batches of inputs with optional noise."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Compute observables for each set of parameters."""
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    row.append(float(val.mean().cpu()))
                results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Wraps :class:`FastBaseEstimator` adding Gaussian shot noise."""

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["Conv", "SamplerQNN", "FastBaseEstimator", "FastEstimator"]
