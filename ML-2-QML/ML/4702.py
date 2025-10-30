"""Hybrid classical classifier combining feed‑forward, self‑attention, and fast estimation."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, Tuple, Callable, List, Sequence

# ----- Self‑attention helper ---------------------------------------------------
class ClassicalSelfAttention(nn.Module):
    """Simple self‑attention block operating on the feature dimension."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        scores = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)
        return torch.matmul(scores, v)

# ----- Fast estimator utilities -----------------------------------------------
class FastBaseEstimator:
    """Evaluates a torch model for many input batches with optional shot noise."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        inputs: np.ndarray,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]] | None = None,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        if observables is None:
            observables = [lambda x: x.mean(dim=-1)]
        obs = list(observables)
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            batch = torch.as_tensor(inputs, dtype=torch.float32)
            outputs = self.model(batch)
            for o in obs:
                val = o(outputs)
                if isinstance(val, torch.Tensor):
                    scalar = float(val.mean().cpu())
                else:
                    scalar = float(val)
                results.append([scalar])
        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy = []
            for row in results:
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                noisy.append(noisy_row)
            return noisy
        return results

# ----- Hybrid classifier --------------------------------------------------------
class HybridClassifier(nn.Module):
    """Feed‑forward network optionally enriched with self‑attention."""
    def __init__(
        self,
        num_features: int,
        depth: int,
        use_attention: bool = True,
        attention_dim: int | None = None,
    ):
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.use_attention = use_attention
        attention_dim = attention_dim or num_features
        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.append(linear)
            layers.append(nn.ReLU())
            if use_attention:
                layers.append(ClassicalSelfAttention(attention_dim))
                in_dim = attention_dim
            else:
                in_dim = num_features
        self.head = nn.Linear(in_dim, 2)
        layers.append(self.head)
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

def build_classifier_circuit(
    num_features: int,
    depth: int,
    use_attention: bool = True,
    attention_dim: int | None = None,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[Callable]]:
    """Return a hybrid classifier and metadata compatible with the original API."""
    model = HybridClassifier(num_features, depth, use_attention, attention_dim)
    encoding = list(range(num_features))
    weight_sizes = [p.numel() for p in model.parameters()]
    observables: List[Callable] = [lambda out: out.mean(dim=-1)]
    return model, encoding, weight_sizes, observables

__all__ = ["HybridClassifier", "build_classifier_circuit", "FastBaseEstimator"]
