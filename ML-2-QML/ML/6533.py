"""HybridClassifier – a classical neural network that mirrors the structure of the quantum ansatz.

The network is depth‑controlled and exposes the same metadata (encoding indices,
parameter counts, and observables) as the quantum counterpart, enabling side‑by‑side
benchmarking.  Observables are user‑supplied callables applied to the network output,
and optional Gaussian shot noise can be added to emulate quantum measurement statistics.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List, Callable
import torch
import torch.nn as nn
import numpy as np

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Iterable[float]) -> torch.Tensor:
    t = torch.as_tensor(list(values), dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t

class HybridClassifier(nn.Module):
    """Feed‑forward network that replicates the depth of the quantum ansatz."""
    def __init__(self, num_features: int, depth: int) -> None:
        super().__init__()
        self.num_features = num_features
        self.depth = depth

        layers: List[nn.Module] = []
        in_dim = num_features
        self.encoding = list(range(num_features))
        self.weight_sizes: List[int] = []

        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.append(linear)
            layers.append(nn.ReLU())
            self.weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = num_features

        head = nn.Linear(in_dim, 2)
        layers.append(head)
        self.weight_sizes.append(head.weight.numel() + head.bias.numel())

        self.network = nn.Sequential(*layers)
        self.default_observables: List[ScalarObservable] = [lambda out: out.mean(dim=-1)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Iterable[Iterable[float]] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Compute observables for a batch of inputs.

        Parameters
        ----------
        observables: Optional[Iterable[ScalarObservable]]
            Callables that map the network output to a scalar.  If omitted,
            the mean over the two output logits is returned.
        parameter_sets: Optional[Iterable[Iterable[float]]]
            Iterable of feature vectors to be fed into the network.
        shots: Optional[int]
            If provided, Gaussian noise with variance 1/shots is added to each
            observable to emulate shot noise.
        seed: Optional[int]
            Seed for the noise generator.
        """
        if observables is None:
            observables = self.default_observables
        if parameter_sets is None:
            raise ValueError("parameter_sets must be provided")
        observables = list(observables)

        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.forward(inputs)
                row = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = val.mean().item()
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy = []
            for row in results:
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                noisy.append(noisy_row)
            return noisy
        return results

    @property
    def total_params(self) -> int:
        return sum(self.weight_sizes)

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[HybridClassifier, Iterable[int], Iterable[int], List[ScalarObservable]]:
    """Construct a feed‑forward classifier and metadata similar to the quantum variant."""
    model = HybridClassifier(num_features, depth)
    encoding = model.encoding
    weight_sizes = model.weight_sizes
    observables = model.default_observables
    return model, encoding, weight_sizes, observables
