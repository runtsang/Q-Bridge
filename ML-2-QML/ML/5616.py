import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable, List, Sequence, Callable

class SamplerModule(nn.Module):
    def __init__(self, bias: bool = False) -> None:
        super().__init__()
        layers = [nn.Linear(2, 4), nn.Tanh()]
        if bias:
            layers.append(nn.Linear(4, 4, bias=True))
        layers.append(nn.Linear(4, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)

class FastEstimator:
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        param_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for vals in param_sets:
                inp = torch.as_tensor(vals, dtype=torch.float32)
                if inp.ndim == 1:
                    inp = inp.unsqueeze(0)
                out = self.model(inp)
                row: List[float] = []
                for obs in observables:
                    val = obs(out)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().cpu())
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)
        return results

    def parameterize(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)

class RegressionHead(nn.Module):
    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

class _ClassicalTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)

class HybridSamplerRegressor(nn.Module):
    def __init__(
        self,
        sampler_bias: bool = False,
        use_qfeatures: bool = False,
        embed_dim: int = 8,
        num_heads: int = 1,
        ffn_dim: int = 16,
        n_layers: int = 1
    ) -> None:
        super().__init__()
        self.sampler = SamplerModule(bias=sampler_bias)
        self.estimator = FastEstimator(self.sampler)
        self.head = RegressionHead()
        self.use_qfeatures = use_qfeatures
        if use_qfeatures:
            self.transformer = nn.Sequential(
                *[
                    _ClassicalTransformerBlock(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        ffn_dim=ffn_dim
                    )
                    for _ in range(n_layers)
                ]
            )
        else:
            self.transformer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.estimator.parameterize(x)
        if self.transformer:
            probs = self.transformer(probs)
        return self.head(probs)
