import torch
import torch.nn as nn
import numpy as np
from typing import Iterable, Sequence, List, Callable

# Lightweight estimator utilities
class _FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables."""
    def __init__(self, model: nn.Module):
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = val.mean().item()
                    row.append(float(val))
                results.append(row)
        return results


# Classical self‑attention helper
class _ClassicalSelfAttention:
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        query = torch.tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key   = torch.tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        value = torch.tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()


# Classical fully‑connected layer mock
class _FullyConnectedLayer(nn.Module):
    def __init__(self, n_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        vals = torch.tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        return torch.tanh(self.linear(vals)).mean().item()


class QuantumNATGen041(nn.Module):
    """Hybrid classical model that mirrors the Quantum‑NAT architecture."""
    def __init__(self) -> None:
        super().__init__()
        # CNN backbone
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Self‑attention block
        self.attn = _ClassicalSelfAttention(embed_dim=4)
        # Fully‑connected mock layer
        self.fcl = _FullyConnectedLayer()
        # Normalisation
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x)
        pooled = torch.nn.functional.avg_pool2d(feat, 6).view(bsz, 16)
        # Classical self‑attention with random params
        rot = np.random.randn(4, 4)
        ent = np.random.randn(4, 4)
        attn_out = self.attn.run(rot, ent, pooled.cpu().numpy())
        # Fully‑connected layer on attention output
        fcl_out = self.fcl.run(attn_out.flatten())
        out = torch.tensor(fcl_out, dtype=torch.float32).unsqueeze(0).repeat(bsz, 1)
        return self.norm(out)

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        estimator = _FastBaseEstimator(self)
        return estimator.evaluate(observables, parameter_sets)


__all__ = ["QuantumNATGen041"]
