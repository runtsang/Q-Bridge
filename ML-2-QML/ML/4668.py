import torch
from torch import nn
import numpy as np

from typing import Iterable, Callable, Sequence, List

from FastBaseEstimator import FastEstimator
from SelfAttention import SelfAttention

class QCNNModel(nn.Module):
    """Hybrid QCNN: classical feature extraction with embedded self‑attention."""
    def __init__(self, feature_dim: int = 8, hidden_dim: int = 16) -> None:
        super().__init__()
        self.attention = SelfAttention()
        self.feature_map = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(hidden_dim // 2, hidden_dim // 2), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(hidden_dim // 4, hidden_dim // 4), nn.Tanh())
        self.head = nn.Linear(hidden_dim // 4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Classical self‑attention feature enhancement
        attn_out = self.attention.run(
            rotation_params=np.random.randn(self.attention.embed_dim, self.attention.embed_dim),
            entangle_params=np.random.randn(self.attention.embed_dim),
            inputs=inputs.detach().cpu().numpy(),
        )
        x = torch.as_tensor(attn_out, dtype=torch.float32)
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

def QCNNML() -> QCNNModel:
    """Factory returning a fully‑configured QCNNModel."""
    return QCNNModel()

class HybridEstimator(FastEstimator):
    """Estimator that evaluates the QCNNModel over batches of parameters."""
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        return super().evaluate(observables, parameter_sets, shots=shots, seed=seed)

__all__ = ["QCNNML", "QCNNModel", "HybridEstimator"]
