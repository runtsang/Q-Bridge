import torch
import numpy as np
from torch import nn
from typing import Callable, Iterable, List, Sequence, Union

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def SamplerQNN() -> nn.Module:
    """A lightweight neural sampler mirroring the Qiskit SamplerQNN."""
    class SamplerModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return torch.softmax(self.net(inputs), dim=-1)

    return SamplerModule()


class FastBaseEstimatorGen:
    """
    Hybrid estimator that can wrap any PyTorch nn.Module (including SamplerQNN)
    and evaluate observables over batches of parameter sets with optional
    shot‑noise simulation.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.model.eval()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Compute scalar observables for each parameter set.  If *shots* is
        provided, Gaussian shot noise is added to each result.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
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


__all__ = ["FastBaseEstimatorGen", "SamplerQNN"]
