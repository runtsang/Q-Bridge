from __future__ import annotations

import torch
from torch import nn
from.Conv import Conv
from.EstimatorQNN import EstimatorQNN

class FraudDetectionHybrid(nn.Module):
    """
    Hybrid fraud detection model that fuses a classical convolutional pre‑processor,
    a lightweight regression network and an externally supplied quantum evaluator.
    The quantum evaluator is injected as a callable, enabling flexible back‑end choice
    (Strawberry Fields, Qiskit, etc.) while keeping the classical module pure NumPy/PyTorch.
    """

    def __init__(self,
                 conv_kernel: int = 2,
                 conv_threshold: float = 0.0,
                 quantum_fn: callable | None = None) -> None:
        super().__init__()
        self.conv = Conv(kernel_size=conv_kernel, threshold=conv_threshold)
        self.estimator = EstimatorQNN()
        self.quantum_fn = quantum_fn

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # Classical path
        conv_out = self.conv.run(data.detach().cpu().numpy())
        est_out = self.estimator(data).item()
        combined = conv_out + est_out

        # Quantum path
        if self.quantum_fn is None:
            raise ValueError("Quantum function not provided")
        q_out = self.quantum_fn(combined)
        return torch.tensor(q_out, dtype=torch.float32)

__all__ = ["FraudDetectionHybrid"]
