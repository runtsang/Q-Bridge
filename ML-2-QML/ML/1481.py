import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingNet(nn.Module):
    """Learned feature extractor that reduces high-dimensional convolutional output to a compact embedding."""
    def __init__(self, in_features: int, embed_dim: int = 8):
        super().__init__()
        self.fc = nn.Linear(in_features, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.fc(x)))

class QuantumHybridClassifier(nn.Module):
    """Extended classical hybrid architecture that mimics the quantum head with a learnable linear layer."""
    def __init__(self, conv_backbone: nn.Module, embed_dim: int = 8):
        super().__init__()
        self.backbone = conv_backbone
        # conv_backbone must expose an `out_features` attribute (e.g. the size of its flattened output)
        self.embed = EmbeddingNet(self.backbone.out_features, embed_dim=embed_dim)
        self.linear = nn.Linear(embed_dim, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        flat = torch.flatten(feats, 1)
        emb = self.embed(flat)
        logits = self.linear(emb)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)
