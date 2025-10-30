import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable, Iterable, List, Sequence

class QuantumNATHybrid(nn.Module):
    """Classical hybrid model that fuses convolutional feature extraction,
    a quantum‑inspired kernel layer, and a fully‑connected classifier.
    Designed to mirror the structure of the original Quantum‑NAT while
    remaining entirely classical, enabling fast prototyping and
    back‑end agnostic evaluation."""
    def __init__(self, num_classes: int = 10, hidden_dim: int = 64) -> None:
        super().__init__()
        # Quanvolution‑style first layer (2×2 kernel, stride 2)
        self.qfilter = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        # Classical convolutional backbone
        self.features = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Quantum‑inspired kernel: fixed random orthogonal transform
        self.qkernel = nn.Linear(16 * 7 * 7, hidden_dim, bias=False)
        with torch.no_grad():
            # initialise with random orthogonal matrix
            W = torch.qr(torch.randn(16 * 7 * 7, hidden_dim))[0]
            self.qkernel.weight.copy_(W)
        # Classifier head
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Extract 2×2 patches via quanvolution filter
        x = self.qfilter(x)
        # Convolutional backbone
        x = self.features(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Quantum‑inspired kernel
        x = self.qkernel(x)
        x = F.relu(x)
        # Classifier
        logits = self.classifier(x)
        return self.norm(logits)

    # ------------------------------------------------------------------
    # Evaluation utilities mirroring FastBaseEstimator
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Evaluate a list of observable callables on batches of inputs."""
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = val.mean().item()
                    row.append(float(val))
                results.append(row)
        return results

__all__ = ["QuantumNATHybrid"]
