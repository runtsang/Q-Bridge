import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ResidualBlock(nn.Module):
    """A simple residual block with two 3x3 convolutions."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride!= 1 or in_channels!= out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out

class QuantumNATGen321(nn.Module):
    """Hybrid classical CNN with residual blocks and a multi‑task head."""
    def __init__(self, num_classes: int = 4, num_regress: int = 2):
        super().__init__()
        # Data‑augmentation pipeline
        self.augment = A.Compose([
            A.RandomCrop(24, 24, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(),
            ToTensorV2()
        ])
        # Backbone: a shallow ResNet‑style CNN
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ResidualBlock(16, 32, stride=2),
            ResidualBlock(32, 64, stride=2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        # Regression head
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_regress)
        )
        self.norm = nn.BatchNorm1d(num_classes + num_regress)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, H, W)
        augmented = []
        for i in range(x.shape[0]):
            # Convert to numpy image for albumentations
            img = x[i].squeeze(0).permute(1, 2, 0).cpu().numpy()
            aug = self.augment(image=img)
            augmented.append(aug["image"])
        x_aug = torch.stack(augmented).to(x.device)
        features = self.backbone(x_aug)
        cls_out = self.classifier(features)
        reg_out = self.regressor(features)
        out = torch.cat([cls_out, reg_out], dim=1)
        return self.norm(out)

__all__ = ["QuantumNATGen321"]
