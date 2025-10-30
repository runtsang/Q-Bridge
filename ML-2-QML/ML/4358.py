import torch
import torch.nn as nn
import numpy as np

class RBFKernel(nn.Module):
    """Classical radial‑basis‑function kernel."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class ClassicalSelfAttention(nn.Module):
    """Drop‑in replacement for the quantum attention block."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        scores = torch.softmax(Q @ K.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ V

class ClassicalClassifier(nn.Module):
    """Feed‑forward classifier mirroring the quantum helper interface."""
    def __init__(self, input_dim: int, depth: int):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, input_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(input_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class QuantumClassifierModel(nn.Module):
    """Hybrid interface that can be instantiated in classical mode."""
    def __init__(self, num_features: int, depth: int, n_qubits: int = 0):
        super().__init__()
        self.lstm = nn.LSTM(num_features, num_features, batch_first=True)
        self.attention = ClassicalSelfAttention(num_features)
        self.classifier = ClassicalClassifier(num_features, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        attn_out = self.attention(last_hidden.unsqueeze(1))
        attn_out = attn_out.squeeze(1)
        logits = self.classifier(attn_out)
        return logits

__all__ = ["QuantumClassifierModel"]
