import torch
import numpy as np
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Union

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Coerce a 1‑D sequence into a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FCL(nn.Module):
    """
    Classical fully‑connected layer with batch evaluation and optional shot noise.
    The interface mirrors the quantum counterpart: `run` evaluates a single
    parameter set, while `evaluate` processes a batch of parameters and
    observables.
    """
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Return the mean tanh of the linear output for a single parameter set."""
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Batch‑evaluate observables over multiple parameter sets.
        If `shots` is provided, Gaussian shot noise with variance 1/shots
        is added to each result.
        """
        if observables is None:
            observables = [lambda out: out.mean(dim=-1)]
        if parameter_sets is None:
            return []

        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self(inputs)
                row: List[float] = []
                for observable in observables:
                    val = observable(outputs)
                    scalar = float(val.mean().cpu()) if isinstance(val, torch.Tensor) else float(val)
                    row.append(scalar)
                results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[float]] = []
            for row in results:
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                noisy.append(noisy_row)
            return noisy

        return results

__all__ = ["FCL"]
