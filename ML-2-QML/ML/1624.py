import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual block with two conv layers and a shortcut."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if in_ch!= out_ch or stride!= 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class QuantumNATEnhanced(nn.Module):
    """
    Classical CNN with residual blocks, dropout, and an auxiliary output.
    Returns a tuple (logits, aux_logits) for compatibility with hybrid loss functions.
    """
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            ResidualBlock(1, 8, stride=1),
            nn.MaxPool2d(2),
            ResidualBlock(8, 16, stride=1),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.2)
        )
        self.fc_main = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
        # Auxiliary classifier after the first residual block
        self.fc_aux = nn.Sequential(
            nn.Linear(8 * 14 * 14, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )
        self.norm_main = nn.BatchNorm1d(num_classes)
        self.norm_aux = nn.BatchNorm1d(num_classes)

    def forward(self, x):
        # Auxiliary path
        aux_feat = self.features[0](x)               # after first residual block
        aux_flat = aux_feat.view(aux_feat.size(0), -1)
        aux_out = self.fc_aux(aux_flat)

        # Main path
        feat = self.features(x)
        flat = feat.view(feat.size(0), -1)
        out = self.fc_main(flat)

        return self.norm_main(out), self.norm_aux(aux_out)

__all__ = ["QuantumNATEnhanced"]
