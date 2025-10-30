import torch
from torch import nn
from typing import Any

class QCNNHybridModel(nn.Module):
    """
    Classical classifier that consumes quantum expectation values produced by a QCNN‑style EstimatorQNN.
    The architecture mimics a convolutional network with fully connected layers and dropout.
    """
    def __init__(self, qnn: Any, input_dim: int = 8) -> None:
        super().__init__()
        self.qnn = qnn
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(16, 12),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(12, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantum expectation values (shape: batch x input_dim)
        q_out = self.qnn.predict(x.detach().cpu().numpy())
        q_out = torch.from_numpy(q_out).float().to(x.device)
        return self.classifier(q_out)

def QCNNHybrid(qnn: Any) -> QCNNHybridModel:
    """Factory returning the configured QCNNHybridModel."""
    return QCNNHybridModel(qnn)

__all__ = ["QCNNHybrid", "QCNNHybridModel"]
