import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence
import numpy as np

@dataclass
class SamplerParams:
    """Container for the two sub‑components of the hybrid sampler."""
    classical: torch.Tensor   # parameters for the classical embedding
    quantum: torch.Tensor    # parameters for the quantum sampler

class ClassicalFilter(nn.Module):
    """A minimal linear embedding that mimics a quanvolution filter."""
    def __init__(self, in_features: int = 2, out_features: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.relu(self.linear(x))

class HybridSamplerQNN(nn.Module):
    """
    Hybrid sampler that first embeds the input through a classical filter
    and then forwards the embedding to a quantum sampler.

    Parameters
    ----------
    quantum_sampler : Callable[[torch.Tensor], torch.Tensor]
        A callable that accepts a 1‑D tensor of quantum parameters and
        returns a probability vector.  The typical implementation is
        the quantum side defined in ``qml_code``.
    in_features : int, optional
        Size of the classical input.
    hidden : int, optional
        Size of the classical embedding.
    """
    def __init__(
        self,
        quantum_sampler: Callable[[torch.Tensor], torch.Tensor],
        in_features: int = 2,
        hidden: int = 4,
    ) -> None:
        super().__init__()
        self.filter = ClassicalFilter(in_features, hidden)
        self.quantum_sampler = quantum_sampler

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the quantum probability vector for a batch of inputs."""
        x = self.filter(inputs)
        return self.quantum_sampler(x)

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Batched evaluation of observables with optional shot noise.

        Parameters
        ----------
        observables : iterable of callables
            Each callable accepts a probability vector and returns a scalar
            or a torch tensor.
        parameter_sets : sequence of parameter sequences
            Each entry contains the full set of parameters (classical + quantum)
            for a single evaluation.
        shots : int, optional
            If provided, Gaussian shot noise with variance 1/shots is added.
        seed : int, optional
            Random seed for the noise generator.

        Returns
        -------
        List[List[float]]
            Nested list where the outer dimension corresponds to the
            parameter sets and the inner dimension to the observables.
        """
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inp = torch.tensor(params, dtype=torch.float32).unsqueeze(0)
                out = self.forward(inp)
                row: List[float] = []
                for obs in observables:
                    val = obs(out)
                    if isinstance(val, torch.Tensor):
                        val = val.mean().item()
                    row.append(float(val))
                results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy = [
                [float(rng.normal(m, max(1e-6, 1 / shots))) for m in row]
                for row in results
            ]
            return noisy
        return results

__all__ = ["HybridSamplerQNN", "SamplerParams"]
