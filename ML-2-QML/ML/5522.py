import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedQuantumNat(nn.Module):
    """Classical hybrid architecture combining CNN, quanvolution and a deep FC head."""
    def __init__(self, num_classes: int = 10, fc_depth: int = 2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.quanv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        cnn_out_dim = 16 * 7 * 7
        quanv_out_dim = 4 * 14 * 14
        fc_input = cnn_out_dim + quanv_out_dim
        layers = []
        in_dim = fc_input
        for _ in range(fc_depth):
            layers.append(nn.Linear(in_dim, 128))
            layers.append(nn.ReLU())
            in_dim = 128
        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
        self.bn = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        cnn_feat = self.cnn(x)
        cnn_flat = cnn_feat.view(bsz, -1)
        quanv_feat = self.quanv(x)
        quanv_flat = quanv_feat.view(bsz, -1)
        features = torch.cat([cnn_flat, quanv_flat], dim=1)
        logits = self.classifier(features)
        return self.bn(logits)

__all__ = ["UnifiedQuantumNat"]
