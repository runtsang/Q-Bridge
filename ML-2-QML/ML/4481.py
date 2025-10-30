import torch
import torch.nn as nn
import numpy as np
from Conv import Conv
from QuantumNAT import QFCModel
from QuantumClassifierModel import build_classifier_circuit

class SelfAttentionGen162:
    """Hybrid classical self‑attention module combining a Quantum‑NAT encoder
    and a feed‑forward classifier.  The module exposes a single ``run`` method
    that accepts a 4‑D tensor (batch, channel, height, width) and returns
    logits for a binary decision task."""
    def __init__(self,
                 embed_dim: int = 4,
                 classifier_depth: int = 2):
        # Classical encoder (inspired by Quantum‑NAT)
        self.encoder = QFCModel()
        # Classical classifier (feed‑forward)
        self.classifier, _, _, _ = build_classifier_circuit(embed_dim, classifier_depth)
        # Simple linear projections for query/key/value
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _attention(self, features: torch.Tensor) -> torch.Tensor:
        """Scaled dot‑product attention on the encoder output."""
        Q = self.query_proj(features)
        K = self.key_proj(features)
        V = self.value_proj(features)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.query_proj.out_features)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V)

    def run(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: 1) encode the image with the Quantum‑NAT style
        convolutional network; 2) apply classical attention; 3) classify.
        """
        # Encode
        encoded = self.encoder(x)  # shape (B, 4)
        # Attention requires a sequence dimension
        seq = encoded.unsqueeze(1)  # shape (B, 1, 4)
        # Attention
        attn_out = self._attention(seq)  # shape (B, 1, 4)
        # Classify
        logits = self.classifier(attn_out.squeeze(1))
        return logits

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.run(x)

__all__ = ["SelfAttentionGen162"]
