import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassicalSelfAttention(nn.Module):
    """Learnable self‑attention block adapted from the quantum version."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.rotation = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (B, L, E)
        query = inputs @ self.rotation
        key = inputs @ self.entangle
        scores = F.softmax(query @ key.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ inputs

def build_classifier_circuit(num_features: int, depth: int = 2):
    """
    Build a classical feed‑forward classifier that mimics the quantum circuit
    interface.  Returns the network, encoding indices, weight sizes, and
    observables.
    """
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

class QuantumClassifierModel(nn.Module):
    """
    CNN backbone + self‑attention + classical dense head.
    Mirrors the structure of QCNet but adds a self‑attention block and
    replaces the quantum head with a classical one.
    """
    def __init__(self, num_features: int, depth: int = 2):
        super().__init__()
        # CNN backbone (same as QCNet)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Self‑attention block
        self.attention = ClassicalSelfAttention(embed_dim=4)

        # Final classifier
        self.classifier, self.enc, self.wts, self.obs = build_classifier_circuit(num_features, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)

        # Dense projection
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Self‑attention on a 4‑dim slice
        attn_input = x[:, :4].unsqueeze(1)          # (B,1,4)
        attn_out = self.attention(attn_input)      # (B,1,4)
        attn_out = attn_out.squeeze(1)             # (B,4)

        # Merge attention with remaining features
        merged = torch.cat([attn_out, x[:, 4:]], dim=1)

        # Final classification head
        logits = self.classifier(merged)
        probs = torch.softmax(logits, dim=-1)
        return probs

__all__ = ["QuantumClassifierModel", "build_classifier_circuit", "ClassicalSelfAttention"]
