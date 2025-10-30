import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from typing import Iterable, List, Sequence, Callable, Union

class HybridSelfAttention(nn.Module):
    """
    Classical hybrid self‑attention that uses a learnable quantum‑derived
    attention matrix.  The quantum parameters are treated as trainable
    tensors and can be optimised jointly with the classical weights.
    """
    def __init__(self, embed_dim: int, seq_len: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        # Classical linear projections
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        # Trainable quantum parameters (encoded as a flat vector)
        self.rotation_params = nn.Parameter(torch.randn(seq_len * 3))
        self.entangle_params = nn.Parameter(torch.randn(seq_len - 1))
        # Optional fraud‑detection head
        self.fraud_head = nn.Linear(seq_len, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, embed_dim)
        """
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        # Classical attention scores
        scores = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1
        )
        # Quantum‑derived attention mask
        quantum_mask = self._quantum_mask(x.shape[0])
        # Hybrid attention
        hybrid_scores = scores * quantum_mask
        out = torch.matmul(hybrid_scores, v)
        return out

    def _quantum_mask(self, batch: int) -> torch.Tensor:
        """
        Convert the rotation and entanglement parameters into a
        positive matrix that can be broadcast over the batch.
        """
        seq = self.seq_len
        mat = torch.nn.functional.normalize(
            self.rotation_params[: seq * seq].view(seq, seq), dim=-1
        )
        mat = mat * mat  # ensure positivity
        return mat.unsqueeze(0).expand(batch, -1, -1)

    # Estimator utilities -------------------------------------------------
    @staticmethod
    def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], Union[torch.Tensor, float]]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Fast deterministic evaluation of the module for a list of
        parameter sets.  Observables are callables that map the module
        output to a scalar (or tensor that can be reduced).
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                # bind parameters
                for i, val in enumerate(params):
                    if i < len(self.rotation_params):
                        self.rotation_params.data[i] = val
                    else:
                        idx = i - len(self.rotation_params)
                        self.entangle_params.data[idx] = val
                # dummy input
                dummy = torch.zeros((1, self.seq_len, self.embed_dim))
                out = self(dummy)
                row: List[float] = []
                for ob in observables:
                    val = ob(out)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        return results

# Auxiliary dataset ---------------------------------------------------------
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# Fraud‑detection helper -----------------------------------------------
def build_fraud_detection_head(input_dim: int, hidden_dim: int = 32) -> nn.Module:
    """
    Simple fully‑connected head that can be appended to the hybrid
    attention output for binary fraud classification.
    """
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, 1),
        nn.Sigmoid(),
    )

__all__ = ["HybridSelfAttention", "RegressionDataset", "generate_superposition_data",
           "build_fraud_detection_head"]
