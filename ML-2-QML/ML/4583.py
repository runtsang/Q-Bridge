import torch
from torch import nn
import numpy as np
from typing import Iterable, List, Sequence, Callable

class HybridFCL(nn.Module):
    """
    Classical fully‑connected network that mirrors the structure of the quantum
    Fully‑Connected Layer (FCL) used in the original seed.  A variable depth
    of linear + ReLU blocks is followed by a classification head.  The
    interface is deliberately identical to the quantum version: a ``run``
    method that accepts a flat list of parameters and returns the network
    output for a unit‑vector input.
    """
    def __init__(self, n_features: int = 1, depth: int = 1, num_classes: int = 2) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = n_features
        for _ in range(depth):
            linear = nn.Linear(in_dim, n_features)
            layers.append(linear)
            layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(n_features, num_classes)
        # Flatten all parameters for quick loading
        self._param_shapes = [p.shape for p in self.parameters()]
        self._param_sizes = [p.numel() for p in self.parameters()]
        self._num_params = sum(self._param_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Load a flattened parameter list into the network, evaluate on a
        unit‑vector input and return the logits as a NumPy array.
        """
        flat = list(thetas)
        if len(flat)!= self._num_params:
            raise ValueError(f"Expected {self._num_params} parameters, got {len(flat)}")
        # Load parameters
        offset = 0
        for shape, size, p in zip(self._param_shapes, self._param_sizes, self.parameters()):
            p.data = torch.tensor(flat[offset:offset + size], dtype=p.dtype).view(shape)
            offset += size

        with torch.no_grad():
            # unit‑vector input: shape (1, n_features)
            inp = torch.ones(1, self.encoder[0].in_features, device=p.device)
            out = self.forward(inp).squeeze()
        return out.cpu().numpy()

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Apply a list of observable callables to the network outputs for each
        parameter set.  Each observable should accept a ``torch.Tensor`` of
        shape (batch, out_features) and return either a scalar tensor or a
        float.  This mirrors the FastBaseEstimator used in the quantum
        version.
        """
        if not observables:
            observables = [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                out = torch.from_numpy(self.run(params)).unsqueeze(0)
                row: List[float] = []
                for obs in observables:
                    val = obs(out)
                    if isinstance(val, torch.Tensor):
                        val = val.mean().item()
                    row.append(float(val))
                results.append(row)
        return results

    @property
    def encoding(self) -> List[int]:
        """Indices of input features that are encoded – trivial for fully‑connected."""
        return list(range(self.encoder[0].in_features))

    @property
    def weight_sizes(self) -> List[int]:
        """Number of parameters per linear block."""
        sizes: List[int] = []
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                sizes.append(layer.weight.numel() + layer.bias.numel())
        sizes.append(self.head.weight.numel() + self.head.bias.numel())
        return sizes

    @property
    def observables(self) -> List[int]:
        """Placeholder to satisfy the quantum interface."""
        return [0] * self.head.out_features

class FastEstimator(HybridFCL):
    """
    Adds optional Gaussian shot noise to the deterministic estimator.
    """
    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
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

__all__ = ["HybridFCL", "FastEstimator"]
