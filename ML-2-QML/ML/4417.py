import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuanvolutionFilter(nn.Module):
    """Classical 2×2 patch convolution used in the quanvolution example."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class SelfAttentionModule(nn.Module):
    """Self‑attention with learnable rotation and entanglement parameters."""
    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.rotation_params = nn.Parameter(torch.randn(embed_dim * 3))
        self.entangle_params = nn.Parameter(torch.randn(embed_dim - 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        query = torch.matmul(inputs, self.rotation_params.reshape(self.embed_dim, -1))
        key = torch.matmul(inputs, self.entangle_params.reshape(self.embed_dim, -1))
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return torch.matmul(scores, inputs)

def build_classifier_head(num_features: int, num_classes: int = 10) -> nn.Module:
    return nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

class HybridNatModel(nn.Module):
    """
    Classical hybrid model combining CNN, quanvolution, self‑attention and a linear head.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Classical convolutional backbone from QuantumNAT
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 4))
        self.norm = nn.BatchNorm1d(4)

        # Quanvolution filter
        self.qfilter = QuanvolutionFilter()

        # Self‑attention module
        self.attention = SelfAttentionModule(embed_dim=4)

        # Classifier head that consumes concatenated features
        total_features = 4 + 784 + 4  # conv_fc + quanv + attention
        self.classifier = build_classifier_head(total_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Classical conv features
        conv_out = self.features(x)
        conv_flat = conv_out.view(bsz, -1)
        conv_fc = self.fc(conv_flat)
        conv_fc = self.norm(conv_fc)

        # Quanvolution features
        quanv = self.qfilter(x)  # shape (bsz, 784)

        # Self‑attention on a small embedding (e.g. mean of conv features)
        attn_input = conv_out.mean(dim=[2, 3])  # (bsz, 16)
        attn_emb = attn_input[:, :4]  # take first 4 dims
        attn_out = self.attention(attn_emb)  # (bsz, 4)

        # Concatenate all
        features = torch.cat([conv_fc, quanv, attn_out], dim=1)
        logits = self.classifier(features)
        return logits

__all__ = ["HybridNatModel"]
