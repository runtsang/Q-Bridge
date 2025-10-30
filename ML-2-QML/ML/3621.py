from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, List, Callable

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Iterable[float]) -> torch.Tensor:
    """Convert a 1‑D iterable of parameters into a 2‑D batch tensor."""
    tensor = torch.as_tensor(list(values), dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridFullyConnectedLayer(nn.Module):
    """
    Classical fully‑connected neural network that mimics the quantum layer interface.

    Parameters
    ----------
    n_features : int
        Number of input features.
    n_outputs : int
        Number of output neurons.
    activation : Callable[[torch.Tensor], torch.Tensor] | None
        Activation applied after the linear map; defaults to ``torch.tanh``.
    """

    def __init__(self, n_features: int = 1, n_outputs: int = 1, activation: Callable[[torch.Tensor], torch.Tensor] | None = torch.tanh) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, n_outputs)
        self.activation = activation

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.linear(inputs)
        return self.activation(x) if self.activation is not None else x

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Mimic the quantum interface: accept a list of parameters and return a numpy array."""
        inputs = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        out = self.forward(inputs)
        return out.detach().cpu().numpy()


class HybridFastEstimator:
    """
    Estimator that evaluates a PyTorch model and optionally adds Gaussian shot noise.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate; typically a ``HybridFullyConnectedLayer``.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: List[List[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Compute observables for each parameter set. If ``shots`` is supplied,
        Gaussian noise with variance 1/shots is added to emulate measurement noise.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    row.append(float(val.mean().cpu()) if isinstance(val, torch.Tensor) else float(val))
                results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["HybridFullyConnectedLayer", "HybridFastEstimator"]
