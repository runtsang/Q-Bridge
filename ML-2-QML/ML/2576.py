import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Hybrid estimator that wraps a PyTorch model, optionally adding shot noise and sampling."""

    def __init__(self, model: nn.Module, *, noise_shots: int | None = None, noise_seed: int | None = None) -> None:
        self.model = model
        self.noise_shots = noise_shots
        self.noise_seed = noise_seed
        self._rng = np.random.default_rng(noise_seed)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
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
        if self.noise_shots is None:
            return results
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [
                float(self._rng.normal(mean, max(1e-6, 1 / self.noise_shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy

    def sample(self, parameter_set: Sequence[float], num_samples: int) -> torch.Tensor:
        """Generate categorical samples from the model's softmax output."""
        self.model.eval()
        with torch.no_grad():
            inputs = _ensure_batch(parameter_set)
            logits = self.model(inputs)
            probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs.squeeze(0), num_samples, replacement=True)


__all__ = ["FastBaseEstimator"]
