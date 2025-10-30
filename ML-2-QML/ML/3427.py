import torch
from torch import nn

class QCNNHybridModel(nn.Module):
    """
    Hybrid classical model that merges a 1‑D QCNN‑style feature extractor with a 2‑D CNN encoder
    from the Quantum‑NAT example.  The two branches are concatenated before the final
    fully‑connected head, allowing the network to process both sequential and spatial
    data within a single forward pass.
    """
    def __init__(self):
        super().__init__()
        # 1‑D QCNN‑style branch (fully‑connected layers mimicking quantum convolution + pooling)
        self.qcnn_branch = nn.Sequential(
            nn.Linear(8, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(),
            nn.Linear(16, 12), nn.Tanh(),
            nn.Linear(12, 8), nn.Tanh(),
            nn.Linear(8, 4), nn.Tanh()
        )
        # 2‑D CNN encoder (Quantum‑NAT style)
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Core head mapping the concatenated features to a 4‑dimensional output
        self.head = nn.Sequential(
            nn.Linear(4 + 16 * 7 * 7, 64), nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accepts either a batch of 1‑D feature vectors (shape [B, 8]) or 2‑D grayscale images
        (shape [B, 1, 28, 28]).  The appropriate branch is activated based on the input
        dimensionality.  The outputs of the two branches are concatenated before the
        final head.
        """
        if x.dim() == 2:  # 1‑D feature input
            a = self.qcnn_branch(x)
            b = torch.zeros(x.size(0), 16 * 7 * 7, device=x.device)
        else:  # 2‑D image input
            a = torch.zeros(x.size(0), 4, device=x.device)
            b = self.cnn_branch(x).view(x.size(0), -1)
        out = torch.cat([a, b], dim=1)
        out = self.head(out)
        return self.norm(out)

def QCNNHybrid() -> QCNNHybridModel:
    """
    Factory returning an instance of the hybrid classical model.
    """
    return QCNNHybridModel()

__all__ = ["QCNNHybridModel", "QCNNHybrid"]
