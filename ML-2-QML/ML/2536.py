import torch
import torch.nn as nn
import numpy as np
from collections.abc import Iterable, Sequence
from typing import Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastBaseEstimator:
    """Hybrid batch estimator for PyTorch models with optional shot‑noise emulation.

    The class extends the original lightweight estimator by:
    * GPU device selection for fast inference.
    * Batched evaluation of many parameter sets in a single forward pass.
    * Optional Gaussian noise that mimics finite‑shot statistics.
    * Support for any callable scalar observable on the model output.
    """

    def __init__(self, model: nn.Module, device: torch.device | str | None = None) -> None:
        self.model = model
        self.device = torch.device(device or "cpu")
        self.model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Return a matrix of observable values for each parameter set.

        Parameters
        ----------
        observables:
            Iterable of callables that map a model output tensor to a scalar.
        parameter_sets:
            Sequence of parameter vectors; each vector is fed as a single input.
        shots:
            If provided, Gaussian noise with variance 1/shots is added to each mean.
        seed:
            Random seed for reproducibility of the noise.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        # Batch all parameters for a single forward pass
        batch = torch.stack([_ensure_batch(p) for p in parameter_sets], dim=0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batch)  # shape (N,...)
            for obs in observables:
                values = obs(outputs)  # shape (N,) or scalar
                if isinstance(values, torch.Tensor):
                    values = values.cpu().numpy()
                else:
                    values = np.array([values] * batch.shape[0])
                results.append(values.tolist())

        # Transpose to match original API: rows per parameter set
        results = [list(row) for row in zip(*results)]

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy = []
            for row in results:
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                noisy.append(noisy_row)
            results = noisy

        return results

# Example hybrid CNN‑quantum model (classical part only)
class QFCModel(nn.Module):
    """Classical convolutional feature extractor used in the Quantum‑NAT example."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 4))
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)

__all__ = ["FastBaseEstimator", "QFCModel"]
