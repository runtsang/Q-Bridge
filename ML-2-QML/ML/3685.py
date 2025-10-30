import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

class SelfAttentionGen330(nn.Module):
    """Hybrid classical self‑attention with estimator behaviour."""
    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> torch.Tensor:
        # Convert to tensors
        rot = torch.from_numpy(rotation_params.reshape(self.embed_dim, -1)).float()
        ent = torch.from_numpy(entangle_params.reshape(self.embed_dim, -1)).float()
        inp = torch.from_numpy(inputs).float()
        # Compute query/key
        query = inp @ rot.T
        key = inp @ ent.T
        # Attention scores
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return scores @ inp

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate the attention block for each parameter set and compute scalar observables.
        Each parameter set must be a tuple (rotation, entangle, inputs).
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                if len(params)!= 3:
                    raise ValueError("Parameter set must contain rotation, entangle, and inputs.")
                rot, ent, inp = params
                out = self.forward(rot, ent, inp)
                row: List[float] = []
                for obs in observables:
                    v = obs(out)
                    if isinstance(v, torch.Tensor):
                        val = float(v.mean().cpu())
                    else:
                        val = float(v)
                    row.append(val)
                results.append(row)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy = []
        for row in results:
            noisy.append([rng.normal(mean, max(1e-6, 1 / shots)) for mean in row])
        return noisy

__all__ = ["SelfAttentionGen330"]
