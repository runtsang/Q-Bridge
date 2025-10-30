import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Callable, Sequence

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of floats to a 2‑D batch tensor."""
    t = torch.as_tensor(values, dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t

class SelfAttentionModule(nn.Module):
    """Classical self‑attention block mirroring the quantum interface."""
    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = F.softmax(q @ k.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v

class SamplerNetwork(nn.Module):
    """Simple probabilistic sampler that can be chained after the head."""
    def __init__(self, inp_dim: int = 2, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_dim, 4),
            nn.Tanh(),
            nn.Linear(4, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)

class QuantumNATGen220(nn.Module):
    """
    Hybrid classical model that integrates a CNN backbone, a self‑attention
    module, and an optional sampler.  Evaluation follows the FastEstimator
    pattern and supports optional shot noise emulation.
    """
    def __init__(self) -> None:
        super().__init__()
        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Attention over pooled features
        self.attn = SelfAttentionModule(embed_dim=16)
        # Linear head to 4‑dimensional output
        self.head = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)
        # Optional sampler for probability outputs
        self.sampler = SamplerNetwork()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feats = self.features(x)                 # [bsz, 16, 14, 14]
        pooled = F.avg_pool2d(feats, 6).view(bsz, -1)  # [bsz, 16*7*7]
        attn_out = self.attn(pooled)             # [bsz, 16*7*7]
        out = self.head(attn_out)                # [bsz, 4]
        out = self.norm(out)
        return out

    def evaluate(
        self,
        observables: Sequence[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate a list of scalar observables for a set of input parameter
        vectors.  The implementation follows the FastEstimator pattern
        but is fully deterministic unless `shots` is provided, in which
        case Gaussian noise with variance 1/shots is added to emulate
        quantum shot noise.
        """
        if not observables:
            observables = [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inp = _ensure_batch(params)
                out = self(inp)
                row: List[float] = []
                for obs in observables:
                    val = obs(out)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().cpu())
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["QuantumNATGen220"]
