import torch
import torch.nn as nn
from.qml import HybridNAT as QuantumHybridNAT

class HybridNAT(nn.Module):
    """Hybrid classical–quantum model merging image and tabular data."""
    def __init__(self, num_classes: int = 4, tabular_input_dim: int = 2):
        super().__init__()
        # Convolutional feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.cnn_fc = nn.Linear(16 * 7 * 7, 64)
        # Tabular data pathway
        self.tabular_fc = nn.Sequential(
            nn.Linear(tabular_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        # Quantum sub‑module that encodes the concatenated embedding
        self.quantum_layer = QuantumHybridNAT(n_wires=4, input_dim=96)
        # Classification head
        self.classifier = nn.Linear(4, num_classes)
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Tensor of shape (B, 1, H, W)
            tabular: Tensor of shape (B, tabular_input_dim)
        Returns:
            Logits of shape (B, num_classes)
        """
        bsz = image.shape[0]
        img_feat = self.cnn(image)
        img_feat = img_feat.view(bsz, -1)
        img_feat = self.cnn_fc(img_feat)
        tab_feat = self.tabular_fc(tabular)
        combined = torch.cat([img_feat, tab_feat], dim=1)
        q_out = self.quantum_layer(combined)
        logits = self.classifier(q_out)
        return self.norm(logits)
