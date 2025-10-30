import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalConvBackbone(nn.Module):
    """Standard 2‑D convolutional feature extractor."""
    def __init__(self, in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size, stride, padding),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
    def forward(self, x):
        return self.features(x)

class ClassicalPatchEncoder(nn.Module):
    """Classical analogue of the quantum patch encoder: flatten 2×2 patches and apply a linear transform."""
    def __init__(self, n_wires=4):
        super().__init__()
        self.n_wires = n_wires
        self.linear = nn.Linear(4, n_wires)
    def forward(self, x):
        bsz = x.shape[0]
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, :, r:r+2, c:c+2]          # shape: (bsz, 1, 2, 2)
                patch = patch.view(bsz, 4)
                out = self.linear(patch)               # shape: (bsz, n_wires)
                patches.append(out)
        return torch.cat(patches, dim=1)                # shape: (bsz, n_wires*14*14)

class LinearFusion(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, x):
        return self.linear(x)

class QuanvolutionHybrid(nn.Module):
    """Classical‑only implementation of the hybrid architecture."""
    def __init__(self, in_channels=1, num_classes=10, n_qubits=4):
        super().__init__()
        self.classical_backbone = ClassicalConvBackbone(in_channels, out_channels=16)
        self.patch_encoder = ClassicalPatchEncoder(n_wires=n_qubits)
        self.fusion = LinearFusion(16*7*7 + n_qubits*14*14, 128)
        self.param_linear = nn.Linear(128, n_qubits)
        self.classifier = nn.Linear(n_qubits, num_classes)

    def forward(self, x):
        class_feat = self.classical_backbone(x).view(x.size(0), -1)
        patch_feat = self.patch_encoder(x)
        fused = torch.cat([class_feat, patch_feat], dim=1)
        fused = self.fusion(fused)
        q_params = self.param_linear(fused)   # classical proxy for quantum parameters
        logits = self.classifier(q_params)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid"]
