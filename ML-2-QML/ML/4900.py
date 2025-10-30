import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassicalQuanvolutionFilter(nn.Module):
    def __init__(self, kernel_size=2, stride=2, out_channels=4, bias=True, threshold=0.0):
        super().__init__()
        self.conv = nn.Conv2d(1, out_channels, kernel_size=kernel_size, stride=stride, bias=bias)
        self.threshold = threshold

    def forward(self, x):
        conv_out = self.conv(x)
        if self.threshold!= 0.0:
            conv_out = torch.sigmoid(conv_out - self.threshold)
        return conv_out.view(x.size(0), -1)

class ClassicalSelfAttention(nn.Module):
    def __init__(self, embed_dim=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.rotation_params = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle_params = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, inputs):
        """
        inputs: (B, N, E)
        """
        query = inputs @ self.rotation_params
        key = inputs @ self.entangle_params
        value = inputs
        scores = F.softmax(query @ key.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ value

class HybridHead(nn.Module):
    def __init__(self, in_features, shift=0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x):
        logits = self.linear(x)
        return torch.sigmoid(logits + self.shift)

class UnifiedQuanvolution(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.filter = ClassicalQuanvolutionFilter()
        self.attention = ClassicalSelfAttention(embed_dim=1)
        self.head = HybridHead(in_features=4 * 14 * 14, shift=0.0)
        self.num_classes = num_classes

    def forward(self, x):
        features = self.filter(x)              # (B, 4*14*14)
        tokens = features.unsqueeze(-1)        # (B, N, 1)
        attn_out = self.attention(tokens)      # (B, N, 1)
        attn_out = attn_out.squeeze(-1)        # (B, N)
        logits = self.head(attn_out)           # (B, 1)
        if self.num_classes == 2:
            probs = torch.cat([logits, 1 - logits], dim=-1)
            return F.log_softmax(probs, dim=-1)
        else:
            return F.log_softmax(logits, dim=-1)

__all__ = ["UnifiedQuanvolution"]
