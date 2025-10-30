import torch
from torch import nn
import numpy as np
from collections.abc import Iterable, Sequence
from typing import Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class HybridEstimator:
    """Hybrid estimator that evaluates a PyTorch model on batches of inputs and observables,
    optionally adding Gaussian shot noise to emulate quantum measurement statistics."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

# --------------------------------------------------------------------------- #
#  Convolution filter (classical) – adapted from pair 2
class ConvFilter(nn.Module):
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[1,2,3])

# --------------------------------------------------------------------------- #
#  Self‑attention helper – adapted from pair 4
class ClassicalSelfAttention:
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

# --------------------------------------------------------------------------- #
#  Hybrid neural network – combines conv, attention and a linear head
class HybridModel(nn.Module):
    def __init__(self, input_dim: int, conv_kernel: int = 2, attention_dim: int = 4):
        super().__init__()
        self.conv = ConvFilter(kernel_size=conv_kernel, threshold=0.0)
        self.attention = ClassicalSelfAttention(embed_dim=attention_dim)
        self.head = nn.Linear(input_dim + attention_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == self.conv.kernel_size ** 2:
            x2d = x.view(-1, 1, self.conv.kernel_size, self.conv.kernel_size)
            conv_out = self.conv(x2d)
            conv_out = conv_out.unsqueeze(-1)
        else:
            conv_out = torch.zeros(x.shape[0], 1, device=x.device)
        attn_out_np = self.attention.run(
            x.numpy(),
            np.random.randn(self.attention.embed_dim, self.attention.embed_dim),
            np.random.randn(self.attention.embed_dim, self.attention.embed_dim),
        )
        attn_out = torch.as_tensor(attn_out_np, dtype=torch.float32)
        combined = torch.cat([conv_out.squeeze(-1), attn_out], dim=-1)
        out = self.head(combined)
        return out.squeeze(-1)

# --------------------------------------------------------------------------- #
#  Regression utilities – adapted from pair 3
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

__all__ = ["HybridEstimator", "HybridModel", "RegressionDataset", "generate_superposition_data"]
