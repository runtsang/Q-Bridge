import torch
from torch import nn
import numpy as np
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

class UnifiedQuantumHybridEstimator:
    """
    A flexible estimator that can evaluate a pure PyTorch model, optionally
    augment it with a quantum head, and optionally add shot‑noise emulation.
    """
    def __init__(self,
                 model: nn.Module,
                 quantum_head: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> None:
        """
        Parameters
        ----------
        model : nn.Module
            A PyTorch module that maps input parameters to raw outputs.
        quantum_head : Callable, optional
            A callable that accepts the model output and returns a scalar
            or tensor representing a quantum‑like expectation. If None,
            the raw model output is used directly.
        """
        self.model = model
        self.quantum_head = quantum_head

    @staticmethod
    def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def evaluate(self,
                 observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
                 parameter_sets: Sequence[Sequence[float]],
                 *,
                 device: Optional[str] = None) -> List[List[float]]:
        """
        Evaluate the model on a list of parameter sets and return a list of
        observable values for each set.
        """
        observables = list(observables) or [lambda out: out.mean()]
        device = device or "cpu"
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = self._ensure_batch(params).to(device)
                outputs = self.model(inputs)

                if self.quantum_head is not None:
                    # Pass the raw outputs through the optional quantum head
                    outputs = self.quantum_head(outputs)

                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = val.mean().item()
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)
        return results

    def evaluate_with_noise(self,
                            observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
                            parameter_sets: Sequence[Sequence[float]],
                            *,
                            shots: int | None = None,
                            seed: int | None = None) -> List[List[float]]:
        """
        Same as ``evaluate`` but optionally adds Gaussian noise to emulate
        shot‑noise when ``shots`` is provided.
        """
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy
