import torch
import torch.nn as nn
import torch.nn.functional as F
from. import QuantumQuanvolutionFilter

class QCNNModel(nn.Module):
    """Classical QCNN-inspired head operating on reduced feature vectors."""
    def __init__(self, input_dim: int = 8) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

class HybridQuanvolutionQCNNClassifier(nn.Module):
    """Hybrid model that merges a quantum quanvolution filter with a classical QCNN head."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuantumQuanvolutionFilter()
        # Reduce the 784‑dimensional quantum feature map to 8 dimensions
        self.reduce = nn.Linear(4 * 14 * 14, 8)
        self.qcnn_head = QCNNModel()
        self.classifier = nn.Linear(1, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qfeat = self.qfilter(x)
        reduced = self.reduce(qfeat)
        qcnn_out = self.qcnn_head(reduced)
        logits = self.classifier(qcnn_out)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQuanvolutionQCNNClassifier", "QCNNModel"]
