import torch
import torch.nn as nn

class QuantumHybridNAT(nn.Module):
    """
    Classical CNN backbone with a random projection head, inspired by QuantumNAT and Quanvolution.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # Classical backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Random projection to 4 features
        self.proj = nn.Linear(16 * 7 * 7, 4)
        # Linear head
        self.fc = nn.Linear(4, num_classes)
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x):
        bsz = x.shape[0]
        features = self.backbone(x)
        flat = features.view(bsz, -1)
        proj = self.proj(flat)
        logits = self.fc(proj)
        return self.norm(logits)
