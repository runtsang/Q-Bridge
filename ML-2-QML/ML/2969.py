import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections.abc import Iterable, Sequence
from typing import Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

class HybridQuanvolution(nn.Module):
    """Classical hybrid model combining a simple 2‑pixel convolution with a
    quantum‑inspired random‑projection kernel and optional shot‑noise
    simulation.  The architecture is intentionally lightweight so that it can
    be benchmarked against its quantum counterpart.
    """
    def __init__(self,
                 num_classes: int = 10,
                 use_noise: bool = False,
                 shots: int | None = None,
                 seed: int | None = None):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        # Random projection mimicking a quantum kernel
        self.kernel = nn.Linear(4 * 14 * 14, 4 * 14 * 14, bias=False)
        nn.init.kaiming_uniform_(self.kernel.weight, a=np.sqrt(5))
        self.linear = nn.Linear(4 * 14 * 14, num_classes)
        self.use_noise = use_noise
        self.shots = shots
        self.rng = np.random.default_rng(seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        flat = features.view(x.size(0), -1)
        projected = self.kernel(flat)
        logits = self.linear(projected)
        return F.log_softmax(logits, dim=-1)

    def evaluate(self,
                 observables: Iterable[ScalarObservable],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        """Evaluate the network for a list of image vectors and observables.
        Each parameter set is interpreted as a flattened 28×28 image.
        """
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                img = torch.tensor(params, dtype=torch.float32).view(1, 1, 28, 28)
                out = self.forward(img)
                row: List[float] = []
                for obs in observables:
                    val = obs(out)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().cpu())
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)
        if self.use_noise and self.shots is not None:
            results = self._add_noise(results)
        return results

    def _add_noise(self, results: List[List[float]]) -> List[List[float]]:
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(self.rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["HybridQuanvolution"]
