import pennylane as qml
import pennylane.numpy as np
import torch
import torch.nn as nn

class QuantumAttentionGate(nn.Module):
    """A lightweight quantum attention module implemented with Pennylane."""
    def __init__(self, n_wires: int = 8):
        super().__init__()
        self.n_wires = n_wires
        self.device = qml.device("default.qubit", wires=n_wires)
        self.qnode = qml.QNode(self._circuit, self.device, interface="torch")

    def _circuit(self, x):
        for i in range(self.n_wires):
            qml.RX(x[i], wires=i)
        for i in range(self.n_wires - 1):
            qml.CNOT(wires=[i, i + 1])
        return torch.stack([qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, seq, dim]
        batch, seq, dim = x.shape
        n = min(self.n_wires, dim)
        out = torch.zeros(batch, seq, self.n_wires, device=x.device)
        for b in range(batch):
            for s in range(seq):
                out[b, s] = self.qnode(x[b, s, :n])
        if self.n_wires < dim:
            pad = torch.zeros(batch, seq, dim - self.n_wires, device=x.device)
            out = torch.cat([out, pad], dim=2)
        return out

class QuantumFeedForwardGate(nn.Module):
    """A lightweight quantum feed‑forward module implemented with Pennylane."""
    def __init__(self, n_wires: int = 8):
        super().__init__()
        self.n_wires = n_wires
        self.device = qml.device("default.qubit", wires=n_wires)
        self.qnode = qml.QNode(self._circuit, self.device, interface="torch")

    def _circuit(self, x):
        for i in range(self.n_wires):
            qml.RY(x[i], wires=i)
        return torch.stack([qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        n = min(self.n_wires, dim)
        out = torch.zeros(batch, seq, self.n_wires, device=x.device)
        for b in range(batch):
            for s in range(seq):
                out[b, s] = self.qnode(x[b, s, :n])
        if self.n_wires < dim:
            pad = torch.zeros(batch, seq, dim - self.n_wires, device=x.device)
            out = torch.cat([out, pad], dim=2)
        return out

class HybridTransformerClassifier(nn.Module):
    """
    Quantum‑only version of the transformer classifier.  All layers are
    implemented with Pennylane circuits.  This is a minimal example that
    demonstrates the API; a full implementation would replace the
    placeholders with proper QNodes.
    """
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int,
                 num_blocks: int, ffn_dim: int, num_classes: int,
                 dropout: float = 0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 5000, embed_dim))
        self.attn_gate = QuantumAttentionGate(n_wires=embed_dim)
        self.ffn_gate = QuantumFeedForwardGate(n_wires=embed_dim)
        self.transformers = nn.ModuleList([self.attn_gate for _ in range(num_blocks)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = tokens + self.pos_embedding[:, :tokens.size(1), :]
        x = self.dropout(x)
        for block in self.transformers:
            x = block(x)
        x = x.mean(dim=1)
        return self.classifier(x)

__all__ = [
    "QuantumAttentionGate",
    "QuantumFeedForwardGate",
    "HybridTransformerClassifier",
]
