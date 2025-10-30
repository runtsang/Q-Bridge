"""Hybrid estimator combining a classical convolutional filter and a lightweight regression neural network, with optional Gaussian shot noise."""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Optional
import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class ConvFilter(nn.Module):
    """Simple 2‑D convolutional pre‑processor that mimics a quantum filter."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        tensor = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

class EstimatorNN(nn.Module):
    """Tiny feed‑forward regression network."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(inputs)

class HybridModel(nn.Module):
    """Chain a convolutional pre‑processor with a regression head."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.pre = ConvFilter(kernel_size, threshold)
        self.reg = EstimatorNN()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs expected shape (batch, 2)
        batch, dims = inputs.shape
        if dims!= 2:
            raise ValueError("HybridModel expects input of shape (batch, 2)")
        # create a 2×2 patch per sample
        patches = inputs.view(batch, 1, 2, 2)
        conv_out = self.pre(patches)
        return self.reg(conv_out)

def build_hybrid_model(kernel_size: int = 2, threshold: float = 0.0) -> nn.Module:
    """Convenience factory returning a ready‑to‑use HybridModel."""
    return HybridModel(kernel_size, threshold)

class FastHybridEstimator:
    """Evaluator that runs a PyTorch model over parameter sets and returns observable values.

    The evaluator is a drop‑in replacement for FastBaseEstimator but can also
    introduce Gaussian shot noise to mimic quantum measurement statistics.
    """
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Compute observable values for each parameter set.

        Parameters
        ----------
        observables
            Iterable of callables that map the model output tensor to a scalar.
        parameter_sets
            Sequence of parameter vectors to feed to the model.
        shots
            If provided, Gaussian noise with variance 1/shots is added.
        seed
            Random seed for reproducibility.
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
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["FastHybridEstimator", "build_hybrid_model", "HybridModel", "ConvFilter", "EstimatorNN"]
