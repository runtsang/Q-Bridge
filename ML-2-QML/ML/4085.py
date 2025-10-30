import torch
from torch import nn
import numpy as np
from collections.abc import Iterable, Sequence
from typing import Callable, List, Union

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastBaseEstimator:
    """Hybrid classical estimator that evaluates any PyTorch model on batched inputs.

    Parameters
    ----------
    model : nn.Module | Callable[[], nn.Module]
        If a factory is supplied it is invoked to create the model.

    Observables are callables that map model outputs to scalars.  If no observables are
    supplied the mean of the raw output is returned.  An optional Gaussian shot‑noise
    can be added to emulate measurement statistics.
    """
    def __init__(self, model: Union[nn.Module, Callable[[], nn.Module]]) -> None:
        if callable(model) and not isinstance(model, nn.Module):
            self.model = model()
        else:
            self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
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
                        val = float(val.mean().cpu())
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy = [[float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                 for row in results]
        return noisy

# Factory helpers -------------------------------------------------------------

def QCNN() -> nn.Module:
    """Return the classical QCNN model used as a drop‑in replacement."""
    from.QCNN import QCNN as _QCNN
    return _QCNN()

def EstimatorQNN() -> nn.Module:
    """Return a lightweight regression network mirroring the Qiskit EstimatorQNN."""
    from.EstimatorQNN import EstimatorQNN as _EstimatorQNN
    return _EstimatorQNN()

__all__ = ["FastBaseEstimator", "QCNN", "EstimatorQNN"]
