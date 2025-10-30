import torch
import numpy as np
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class ClassicalSelfAttention(nn.Module):
    """Classical self‑attention layer mirroring the quantum interface."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, inputs: torch.Tensor,
                rotation_params: np.ndarray,
                entangle_params: np.ndarray) -> torch.Tensor:
        query = inputs @ rotation_params.reshape(self.embed_dim, -1)
        key   = inputs @ entangle_params.reshape(self.embed_dim, -1)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return scores @ inputs

class FastHybridEstimator:
    """Deterministic estimator that can optionally prepend a classical self‑attention block."""
    def __init__(self,
                 model: nn.Module,
                 *,
                 attention: bool = False,
                 attention_embed_dim: int | None = None,
                 rotation_params: np.ndarray | None = None,
                 entangle_params: np.ndarray | None = None,
                 noise_shots: int | None = None,
                 noise_seed: int | None = None) -> None:
        self.model = model
        self.noise_shots = noise_shots
        self.noise_seed = noise_seed

        if attention:
            if attention_embed_dim is None:
                attention_embed_dim = 4
            self.attention = ClassicalSelfAttention(attention_embed_dim)
            self.rotation_params = rotation_params if rotation_params is not None else np.random.rand(attention_embed_dim * 3)
            self.entangle_params = entangle_params if entangle_params is not None else np.random.rand(attention_embed_dim - 1)
        else:
            self.attention = None

    def evaluate(self,
                 observables: Iterable[ScalarObservable],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                if self.attention is not None:
                    inputs = self.attention(inputs,
                                            self.rotation_params,
                                            self.entangle_params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    val = observable(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)

        if self.noise_shots is None:
            return results
        rng = np.random.default_rng(self.noise_seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / self.noise_shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["FastHybridEstimator"]
