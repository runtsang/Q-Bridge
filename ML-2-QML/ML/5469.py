import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuantumNATHybridCore(nn.Module):
    """
    Classical core of the hybrid QuantumNAT model.
    The quantum part is injected via the `quantum_module` argument.
    """
    def __init__(self, quantum_module, embed_dim=4, shift=np.pi/2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Flatten and linear to embed_dim
        self.proj = nn.Linear(16 * 7 * 7, embed_dim)
        self.quantum_module = quantum_module
        # Hybrid head
        self.head = nn.Linear(embed_dim, 1)
        self.shift = shift

    def forward(self, x):
        features = self.backbone(x)
        flat = features.view(features.size(0), -1)
        embed = F.relu(self.proj(flat))
        quantum_out = self.quantum_module(embed)
        logits = self.head(quantum_out)
        probs = torch.sigmoid(logits + self.shift)
        return torch.cat((probs, 1 - probs), dim=-1)
