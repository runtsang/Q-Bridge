import numpy as np
import torch
import torch.nn as nn
from typing import Iterable, List, Callable, Sequence

class ConvFilter(nn.Module):
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, bias: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=bias)

    def run(self, data: np.ndarray) -> float:
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

class FraudGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.activation = nn.Tanh()
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("shift", torch.tensor(0.0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.linear(inputs.unsqueeze(-1)))
        return x * self.scale + self.shift

class ClassicalSelfAttention:
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

class FastEstimator:
    def __init__(self, model: Callable):
        self.model = model

    def evaluate(self, observables: Iterable[Callable[[np.ndarray], float]], parameter_sets: Sequence[Sequence]) -> List[List[float]]:
        results: List[List[float]] = []
        for params in parameter_sets:
            outputs = self.model(*params)
            row = [obs(outputs) for obs in observables]
            results.append(row)
        return results

class SelfAttention:
    def __init__(self, embed_dim: int = 4, conv_kernel: int = 2, conv_threshold: float = 0.0, use_gate: bool = False):
        self.embed_dim = embed_dim
        self.conv = ConvFilter(kernel_size=conv_kernel, threshold=conv_threshold) if conv_kernel else None
        self.gate = FraudGate() if use_gate else None
        self.attention = ClassicalSelfAttention(embed_dim=embed_dim)
        self.estimator = FastEstimator(self.run)

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        if self.conv:
            processed = []
            for row in inputs:
                data = row.reshape(1, 1, self.conv.kernel_size, self.conv.kernel_size)
                processed.append(self.conv.run(data))
            inputs = np.array(processed)
        out = self.attention.run(rotation_params, entangle_params, inputs)
        if self.gate:
            out = self.gate(torch.as_tensor(out, dtype=torch.float32)).numpy()
        return out

    def evaluate(self, observables: Iterable[Callable[[np.ndarray], float]], parameter_sets: Sequence[Sequence]) -> List[List[float]]:
        return self.estimator.evaluate(observables, parameter_sets)

__all__ = ["SelfAttention"]
