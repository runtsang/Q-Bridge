"""Enhanced lightweight estimator utilities built on PyTorch with batched inference,
automatic device placement, and checkpointing support."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> Tensor:
    """Return a 2‑D tensor of shape (batch,...) for uniform processing."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class ParameterDataset(Dataset):
    """Dataset wrapping a list of parameter vectors."""
    def __init__(self, param_list: Sequence[Sequence[float]]):
        self._data = [torch.as_tensor(p, dtype=torch.float32) for p in param_list]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Tensor:
        return self._data[idx]

# --------------------------------------------------------------------------- #
#  Core
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """
    Evaluate a neural network for batches of inputs and observables.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to evaluate.  The model must accept a 2‑D tensor of
        shape (batch, features) and return a 2‑D tensor of shape
        (batch, output_features).
    device : str | torch.device, optional
        Device on which to run the model.  If ``None`` the default device
        determined by ``torch.device('cpu')`` or ``torch.device('cuda')`` if
        available.
    batch_size : int, optional
        Batch size used when evaluating a large number of parameter sets.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        batch_size: int = 128,
    ) -> None:
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.batch_size = batch_size

    def load_state_dict(self, path: str) -> None:
        """Load a checkpoint into the underlying model."""
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Evaluate the model for each parameter set and observable.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the model output tensor and returns a
            scalar (tensor or float) representing the observable.
        parameter_sets : sequence of parameter vectors
            Each vector is a 1‑D sequence of floats.

        Returns
        -------
        List[List[float]]
            A list of rows, one per parameter set, each containing the
            observable values.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            dataset = ParameterDataset(parameter_sets)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            for batch in loader:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                for obs in observables:
                    values = obs(outputs)
                    if isinstance(values, Tensor):
                        values = values.cpu().numpy()
                    else:
                        values = np.array(values)
                    # Flatten and convert to list
                    flat = values.reshape(-1).tolist()
                    # Append each row value
                    for i, val in enumerate(flat):
                        if len(results) <= i:
                            results.append([])
                        results[i].append(val)
        return results

class FastEstimator(FastBaseEstimator):
    """
    Adds optional Gaussian shot noise to the deterministic estimator.

    Parameters
    ----------
    shots : int | None, optional
        Number of shots to simulate.  If ``None`` (default), no noise is added.
    seed : int | None, optional
        Random seed for the noise generator.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        batch_size: int = 128,
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(model, device=device, batch_size=batch_size)
        self.shots = shots
        self.seed = seed

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if self.shots is None:
            return raw
        rng = np.random.default_rng(self.seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["FastBaseEstimator", "FastEstimator"]
