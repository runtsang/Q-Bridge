import torch
import torch.nn as nn
import numpy as np
from quantum_module import FraudDetectionQuantumKernel

class FraudDetectionHybrid(nn.Module):
    """
    Hybrid fraud detection model that merges a classical convolutional
    feature extractor with a quantum kernel based on EstimatorQNN and
    FastEstimator. The model is fully compatible with PyTorch pipelines.
    """
    def __init__(self, quantum_kernel: FraudDetectionQuantumKernel | None = None):
        super().__init__()
        # Classical feature extractor inspired by Quanvolution
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        self.activation = nn.Tanh()
        # Quantum kernel
        self.quantum_kernel = quantum_kernel or FraudDetectionQuantumKernel()
        # Linear head
        self.linear = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 2)
        x = x.view(-1, 1, 1, 2)
        x = self.conv(x)
        x = self.activation(x)
        x = x.view(-1, 4)
        # Evaluate quantum kernel
        q_features = self.quantum_kernel.evaluate(x.detach().cpu().numpy())
        q_features = torch.from_numpy(q_features).to(x.device).float()
        out = self.linear(q_features)
        return out

__all__ = ["FraudDetectionHybrid"]
