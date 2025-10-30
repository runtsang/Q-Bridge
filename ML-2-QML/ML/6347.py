import torch
from torch import nn
import numpy as np
from typing import Iterable, List, Callable, Sequence

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FCL(nn.Module):
    """Classical fully‑connected layer with batched evaluation and optional shot noise."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        return torch.tanh(self.linear(values))

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Return the mean tanh output for a single parameter set."""
        exp = self.forward(thetas).mean(dim=0)
        return exp.detach().cpu().numpy()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Batched evaluation of observables over many parameter sets.

        Parameters
        ----------
        observables : Iterable[Callable[[Tensor], Tensor | float]]
            Functions that map the layer output to a scalar.  If ``None`` a single
            observable that returns the mean of the output is used.
        parameter_sets : Sequence[Sequence[float]]
            A list of parameter vectors, one per evaluation.
        shots : int, optional
            If provided, Gaussian noise with variance 1/shots is added to each
            observable value to mimic shot noise.
        seed : int, optional
            Random seed for reproducible noise.

        Returns
        -------
        List[List[float]]
            A 2‑D list where each row corresponds to a parameter set and each
            column to an observable.
        """
        if parameter_sets is None:
            return []

        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.forward(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    scalar = float(value.mean().cpu()) if isinstance(value, torch.Tensor) else float(value)
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

__all__ = ["FCL"]
