import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float], device: str = "cpu") -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model for batches of parameters and observables.

    Supports optional GPU execution, automatic differentiation, and shot‑noise
    simulation.  The class is deliberately lightweight to be used in research
    pipelines where many evaluations are required.
    """

    def __init__(self, model: nn.Module, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.device = device

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Deterministic evaluation of observables."""
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params, device=self.device)
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

    def evaluate_with_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[tuple[List[float], List[np.ndarray]]]:
        """Return value and gradient for each parameter set."""
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[tuple[List[float], List[np.ndarray]]] = []
        self.model.train()
        for params in parameter_sets:
            inputs = _ensure_batch(params, device=self.device)
            inputs.requires_grad_(True)
            outputs = self.model(inputs)
            row_values: List[float] = []
            row_grads: List[np.ndarray] = []
            for obs in observables:
                val = obs(outputs)
                if isinstance(val, torch.Tensor):
                    val = val.mean()
                else:
                    val = torch.tensor(val, dtype=torch.float32, device=self.device)
                val.backward(retain_graph=True)
                grad = inputs.grad.detach().cpu().numpy().squeeze()
                row_values.append(float(val.cpu()))
                row_grads.append(grad.copy())
                inputs.grad.zero_()
            results.append((row_values, row_grads))
        return results

    def add_shot_noise(
        self,
        outputs: List[List[float]],
        shots: int,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """Add Gaussian shot noise to deterministic outputs."""
        rng = np.random.default_rng(seed)
        noisy = []
        for row in outputs:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator"]
