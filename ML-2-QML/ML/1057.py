import torch
from torch import nn
import numpy as np
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float], device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastBaseEstimator:
    """
    Lightweight estimator for PyTorch models that supports GPU inference,
    optional dropout, and vectorised evaluation of arbitrary scalar observables.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str = "cpu",
        dropout_prob: Optional[float] = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.dropout_prob = dropout_prob

    def _forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.model(inputs)
        if self.dropout_prob is not None:
            outputs = nn.functional.dropout(outputs, p=self.dropout_prob, training=False)
        return outputs

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Compute the mean of each observable over a batch of inputs.
        """
        observables = list(observables) or [lambda x: x.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params, self.device)
                outputs = self._forward(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results

    def compute_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[torch.Tensor]]:
        """
        Return the gradient of each observable with respect to the input parameters.
        """
        observables = list(observables) or [lambda x: x.mean(dim=-1)]
        grads: List[List[torch.Tensor]] = []
        self.model.eval()
        for params in parameter_sets:
            inputs = _ensure_batch(params, self.device)
            inputs.requires_grad_(True)
            outputs = self._forward(inputs)
            row_grads: List[torch.Tensor] = []
            for observable in observables:
                value = observable(outputs)
                if not isinstance(value, torch.Tensor):
                    raise TypeError("Observable must return a torch.Tensor for gradient computation.")
                value.backward(retain_graph=True)
                row_grads.append(inputs.grad.clone())
                inputs.grad.zero_()
            grads.append(row_grads)
        return grads

class FastEstimator(FastBaseEstimator):
    """
    Adds optional Gaussian shot noise to the deterministic estimator.
    """

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
