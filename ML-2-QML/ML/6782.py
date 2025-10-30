import torch
from torch import nn
from typing import Optional

class QCNNBlock(nn.Module):
    """Single classical block that emulates a QCNN convolution + pooling step."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.feature_map = nn.Linear(in_features, 2 * in_features)
        self.conv = nn.Linear(in_features, out_features)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fm = self.feature_map(x)
        left, right = fm.chunk(2, dim=-1)
        conv_out = self.conv(left)
        pooled = self.activation((conv_out + right) / 2)
        return pooled

class HybridHead(nn.Module):
    """Classical head that mimics a quantum expectation layer."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x).squeeze(-1)
        return torch.sigmoid(logits + self.shift)

class UnifiedQCNNHybrid(nn.Module):
    """Hybrid QCNN that stacks classical QCNN blocks and optionally uses a quantum head."""
    def __init__(self,
                 in_features: int,
                 hidden_features: int = 32,
                 num_blocks: int = 3,
                 use_quantum_head: bool = False,
                 quantum_head: Optional[nn.Module] = None):
        super().__init__()
        self.blocks = nn.ModuleList()
        current = in_features
        for _ in range(num_blocks):
            self.blocks.append(QCNNBlock(current, hidden_features))
            current = hidden_features
        self.classifier = nn.Linear(current, 1)
        if use_quantum_head:
            self.head = quantum_head or HybridHead(current)
        else:
            self.head = HybridHead(current)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.classifier(x)
        probs = self.head(x)
        return torch.cat([probs.unsqueeze(-1), (1 - probs).unsqueeze(-1)], dim=-1)

    def freeze_classical(self) -> None:
        """Freeze all classical parameters."""
        for param in self.parameters():
            param.requires_grad = False
        if hasattr(self.head, "parameters"):
            for p in self.head.parameters():
                p.requires_grad = True

    def unfreeze_all(self) -> None:
        """Make all parameters trainable."""
        for p in self.parameters():
            p.requires_grad = True

__all__ = ["QCNNBlock", "HybridHead", "UnifiedQCNNHybrid"]
