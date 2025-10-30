import torch
from torch import nn
import numpy as np
from collections.abc import Iterable, Sequence
from typing import Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a list of scalar parameters into a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastBaseEstimator:
    """Evaluate a neural network over batches of inputs and per‑output observables."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
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
        return results

class FastEstimator(FastBaseEstimator):
    """Add Gaussian shot‑like noise to the deterministic estimator."""
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
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

def EstimatorQNN(
    layer_sizes: Sequence[int] = (2, 8, 4, 1),
    act_fns: Sequence[nn.Module] = (nn.Tanh(), nn.Tanh()),
    dropout: float | None = None,
) -> nn.Module:
    """Instantiate a configurable feed‑forward regression network.

    The network is built from ``layer_sizes`` and ``act_fns``.  An optional
    dropout layer can be inserted after each activation to regularise the
    model.  The returned module can be wrapped by :class:`FastBaseEstimator`
    or :class:`FastEstimator` for batched evaluation.
    """
    layers: List[nn.Module] = []
    in_dim = layer_sizes[0]
    for out_dim, act in zip(layer_sizes[1:-1], act_fns):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(act)
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        in_dim = out_dim
    layers.append(nn.Linear(in_dim, layer_sizes[-1]))
    return nn.Sequential(*layers)

__all__ = ["EstimatorQNN", "FastBaseEstimator", "FastEstimator"]
