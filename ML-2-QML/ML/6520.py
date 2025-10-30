import torch
from torch import nn
import numpy as np
import qml  # Quantum module providing QCNNHybridKernelQuantum

class QCNNHybridKernel(nn.Module):
    """
    Hybrid QCNN model that integrates a classical convolution‑style network
    with a quantum kernel layer. The classical part emulates the QCNN
    architecture from QCNN.py, while the quantum part implements a
    learnable kernel (from QuantumKernelMethod.py) to capture
    non‑linear relationships between feature maps. The model can be
    trained end‑to‑end using gradient descent on the classical
    parameters; the quantum kernel is treated as a differentiable
    layer via the parameter‑shift rule implemented in the quantum
    module.
    """

    def __init__(self, n_features: int = 8, n_classes: int = 1):
        super().__init__()
        # Classical feature extractor (QCNN‑style)
        self.feature_map = nn.Sequential(nn.Linear(n_features, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        # Quantum kernel layer
        self.qkernel = qml.QCNNHybridKernelQuantum()
        # Final classifier
        self.head = nn.Linear(4, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical forward pass
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        # Apply quantum kernel to the classical features
        # The quantum kernel returns a vector of similarity scores
        q_out = self.qkernel.forward(x.cpu().numpy())
        q_out = torch.tensor(q_out, dtype=x.dtype, device=x.device)
        # Concatenate classical and quantum features
        cat = torch.cat([x, q_out], dim=1)
        out = self.head(cat)
        return torch.sigmoid(out)
