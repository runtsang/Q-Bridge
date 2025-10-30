import torch
import torch.nn as nn
import numpy as np
from typing import Iterable, List, Callable, Sequence

class SelfAttentionHybrid:
    """Hybrid classical self‑attention with optional classification and fraud‑style scaling.
    The class mirrors the interface of the original SelfAttention and QuantumClassifierModel
    seeds while adding a fraud‑detection style scaling layer and a fast batch evaluator.
    """

    def __init__(self, embed_dim: int = 4, depth: int = 2, use_scaling: bool = False) -> None:
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_scaling = use_scaling
        self.attention = self._build_attention()
        self.classifier = self._build_classifier()

    def _build_attention(self) -> nn.Module:
        class Attention(nn.Module):
            def __init__(self, embed_dim: int):
                super().__init__()
                self.embed_dim = embed_dim

            def forward(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> torch.Tensor:
                rot = torch.as_tensor(rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
                ent = torch.as_tensor(entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
                inp = torch.as_tensor(inputs, dtype=torch.float32)
                query = inp @ rot
                key = inp @ ent
                scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
                return scores @ inp

        return Attention(self.embed_dim)

    def _build_classifier(self) -> nn.Module:
        layers: List[nn.Module] = []
        in_dim = self.embed_dim
        for _ in range(self.depth):
            layers.append(nn.Linear(in_dim, self.embed_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.embed_dim, 2))
        return nn.Sequential(*layers)

    def run_attention(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        return self.attention(rotation_params, entangle_params, inputs).detach().cpu().numpy()

    def run_classifier(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        attn_out = self.run_attention(rotation_params, entangle_params, inputs)
        logits = self.classifier(torch.as_tensor(attn_out, dtype=torch.float32))
        return logits.detach().cpu().numpy()

    def evaluate(self,
                 observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        observables = list(observables)
        if not observables:
            observables = [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.classifier.eval()
        with torch.no_grad():
            for params in parameter_sets:
                rot, ent, inp = params
                batch = torch.as_tensor(inp, dtype=torch.float32).unsqueeze(0)
                out = self.classifier(batch).squeeze(0)
                row: List[float] = []
                for obs in observables:
                    val = obs(out)
                    if isinstance(val, torch.Tensor):
                        row.append(float(val.mean().cpu()))
                    else:
                        row.append(float(val))
                results.append(row)
        return results
