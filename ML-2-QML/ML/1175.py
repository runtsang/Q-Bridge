import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNATHybrid(nn.Module):
    """
    Classical branch of the hybrid QuantumNAT model.
    Implements a residual CNN with a learnable feature weighting layer
    before the final fully‑connected projection.
    """
    def __init__(self) -> None:
        super().__init__()
        # Residual block 1
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.skip1 = nn.Conv2d(1, 8, kernel_size=1)
        self.pool1 = nn.MaxPool2d(2)
        # Residual block 2
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2)
        # Feature weighting
        self.feature_weight = nn.Linear(16 * 7 * 7, 16 * 7 * 7)
        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual block 1
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.skip1(x)
        out = F.relu(out)
        out = self.pool1(out)
        # Residual block 2
        out = F.relu(self.bn3(self.conv3(out)))
        out = self.bn4(self.conv4(out))
        out = F.relu(out)
        out = self.pool2(out)
        # Flatten
        out = out.view(out.size(0), -1)
        # Feature weighting
        weighted = self.feature_weight(out)
        weighted = torch.sigmoid(weighted) * out
        # Fully connected
        out = self.fc(weighted)
        return self.norm(out)
