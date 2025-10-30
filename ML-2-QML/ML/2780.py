import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the quantum feature extractor defined in the QML module
from EstimatorQNN__gen033_qml import quantum_kernel

class QuanvolutionFilter(nn.Module):
    """
    Classical 2‑D convolutional layer that mimics the original quanvolution filter.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the convolution and flatten the output.
        """
        features = self.conv(x)
        return features.view(x.size(0), -1)

class HybridEstimator(nn.Module):
    """
    Hybrid regressor that concatenates classical quanvolutional features
    with quantum‑kernel embeddings before a final linear head.
    """
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        # Dimension of quantum feature vector (4 qubits → 4 expectation values)
        self.q_feature_dim = 4
        self.linear = nn.Linear(self.q_feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: classical conv → quantum kernel → linear regression.
        """
        # Classical feature extraction
        classical_features = self.qfilter(x)  # shape: (batch, 4*14*14)
        # Select first 4 elements to feed into the quantum kernel
        q_inputs = classical_features[:, :self.q_feature_dim]
        # Quantum feature map (returns tensor of shape (batch, 4))
        q_features = quantum_kernel(q_inputs)
        # Linear head for regression
        logits = self.linear(q_features)
        return logits

def EstimatorQNN() -> nn.Module:
    """
    Factory function that returns the hybrid estimator.
    """
    return HybridEstimator()

__all__ = ["QuanvolutionFilter", "HybridEstimator", "EstimatorQNN"]
