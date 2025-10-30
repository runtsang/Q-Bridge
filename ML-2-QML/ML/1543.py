import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridHead(nn.Module):
    """Dense head producing a probability."""
    def __init__(self, in_features: int, out_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))

class QuantumHybridBinaryClassifier(nn.Module):
    """Classical hybrid binary classifier with optional quantum fallback."""
    def __init__(self,
                 in_features: int,
                 use_quantum: bool = False,
                 quantum_feature_dim: int = 0):
        super().__init__()
        self.use_quantum = use_quantum
        self.quantum_feature_dim = quantum_feature_dim
        total_in = in_features + (quantum_feature_dim if use_quantum else 0)
        self.head = HybridHead(total_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_quantum and self.quantum_feature_dim > 0:
            # Placeholder for quantum features; real implementation would compute them.
            qfeat = torch.zeros((x.size(0), self.quantum_feature_dim), device=x.device)
            x = torch.cat([x, qfeat], dim=-1)
        return self.head(x)

__all__ = ["QuantumHybridBinaryClassifier"]
