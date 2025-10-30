"""QuanvolutionHybrid: Quantum‑centric implementation.

The quantum module mirrors the hybrid architecture but replaces the
classical convolutional backbone with a quantum kernel that
extracts 2×2 patches.  The fully‑connected layer is a parameterized
quantum circuit that outputs an expectation value.  The transformer
classifier uses a quantum‑enhanced attention head and a quantum
feed‑forward network.  The code is written for the Pennylane device
(default.qubit).  All functions are importable as a single module.
"""

from __future__ import annotations

import math
from typing import Optional

import pennylane as qml
import pennylane.numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 1. Quantum patch extraction (Quan‑like)
# --------------------------------------------------------------------------- #
class QuantumQuanvolutionFilter(nn.Module):
    """Quantum‑based patch extraction using a small parameterised circuit.

    The circuit encodes the 4 pixel values of a 2×2 patch into rotation
    angles of 4 qubits, applies a random entangling layer, and measures
    the expectation of Pauli‑Z on each qubit.  The result is a 4‑dim
    feature vector per patch.
    """
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.dev = qml.device("default.qubit", wires=n_wires, shots=200)
        self.circuit = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, x: np.ndarray) -> np.ndarray:
        # x has shape (n_wires,)
        for i, val in enumerate(x):
            qml.RY(val, wires=i)
        # Random entangling layer
        for i in range(self.n_wires - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[self.n_wires - 1, 0])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 4)  flattened patch
        batch = x.shape[0]
        patches = x.view(batch, -1)
        feats = []
        for patch in patches:
            feat = self.circuit(patch.numpy())
            feats.append(torch.tensor(feat, dtype=x.dtype, device=x.device))
        return torch.stack(feats, dim=0)  # (batch, 4)

# --------------------------------------------------------------------------- #
# 2. Quantum fully‑connected layer
# --------------------------------------------------------------------------- #
class QuantumFCLayer(nn.Module):
    """Parameterised quantum circuit that outputs a single expectation value.

    The circuit consists of a single qubit with a trainable Ry gate
    followed by measurement of Pauli‑Z.  The expectation value is
    returned as a 1‑D tensor.
    """
    def __init__(self, n_wires: int = 1, shots: int = 200) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.shots = shots
        self.dev = qml.device("default.qubit", wires=n_wires, shots=shots)
        self.circuit = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, x: np.ndarray) -> np.ndarray:
        for i, val in enumerate(x):
            qml.RY(val, wires=i)
        return qml.expval(qml.PauliZ(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_wires)
        batch = x.shape[0]
        outputs = []
        for sample in x:
            out = self.circuit(sample.numpy())
            outputs.append(torch.tensor(out, dtype=x.dtype, device=x.device))
        return torch.stack(outputs, dim=0).unsqueeze(-1)  # (batch, 1)

# --------------------------------------------------------------------------- #
# 3. Quantum‑enhanced attention head
# --------------------------------------------------------------------------- #
class QuantumAttentionHead(nn.Module):
    """Multi‑head attention where the key/value projections are obtained
    from a small quantum circuit.  The projection uses a single qubit
    per head and a trainable Ry gate.  The attention scores are
    computed classically after the quantum projections.
    """
    def __init__(self, embed_dim: int, num_heads: int, n_wires: int = 1, shots: int = 200) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.n_wires = n_wires
        self.shots = shots
        self.dev = qml.device("default.qubit", wires=n_wires, shots=shots)
        self.circuit = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, x: np.ndarray) -> np.ndarray:
        for i, val in enumerate(x):
            qml.RY(val, wires=i)
        return qml.expval(qml.PauliZ(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, embed_dim)
        batch, seq, _ = x.shape
        # Project each token to d_k dimensions using quantum circuits
        proj = []
        for token in x.unbind(dim=1):  # iterate over sequence
            # split token into heads
            token_heads = token.view(self.num_heads, self.d_k)
            head_outs = []
            for head in token_heads:
                out = self.circuit(head.numpy())
                head_outs.append(torch.tensor(out, dtype=x.dtype, device=x.device))
            proj.append(torch.stack(head_outs, dim=0))
        proj = torch.stack(proj, dim=1)  # (batch, seq, num_heads, d_k)
        # Classical attention using projected keys/values
        q = k = v = proj
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return out.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)

# --------------------------------------------------------------------------- #
# 4. Quantum feed‑forward network
# --------------------------------------------------------------------------- #
class QuantumFeedForward(nn.Module):
    """Two‑layer feed‑forward network realised by a quantum circuit.

    The first layer maps the input qubits to an intermediate space
    using a trainable Ry gate on each qubit.  The second layer maps
    the intermediate space back to the embedding dimension.
    """
    def __init__(self, embed_dim: int, ffn_dim: int, n_wires: int = 4, shots: int = 200) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.n_wires = n_wires
        self.shots = shots
        self.dev = qml.device("default.qubit", wires=n_wires, shots=shots)
        self.circuit = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, x: np.ndarray) -> np.ndarray:
        # x is of shape (n_wires,)
        for i, val in enumerate(x):
            qml.RY(val, wires=i)
        # Entangling layer
        for i in range(self.n_wires - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[self.n_wires - 1, 0])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        outputs = []
        for sample in x:
            out = self.circuit(sample.numpy())
            outputs.append(torch.tensor(out, dtype=x.dtype, device=x.device))
        return torch.stack(outputs, dim=0)

# --------------------------------------------------------------------------- #
# 5. Transformer block using quantum modules
# --------------------------------------------------------------------------- #
class QuantumTransformerBlock(nn.Module):
    """Transformer block that uses quantum attention and quantum feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 n_wires_attention: int = 1, n_wires_ffn: int = 4, shots: int = 200) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.attn = QuantumAttentionHead(embed_dim, num_heads, n_wires_attention, shots)
        self.ffn = QuantumFeedForward(embed_dim, ffn_dim, n_wires_ffn, shots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# --------------------------------------------------------------------------- #
# 6. Positional encoding
# --------------------------------------------------------------------------- #
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

# --------------------------------------------------------------------------- #
# 7. The hybrid quantum model
# --------------------------------------------------------------------------- #
class QuanvolutionHybrid(nn.Module):
    """Quantum‑centric hybrid model that chains a quantum patch extractor,
    a quantum fully‑connected layer, and a transformer classifier that
    uses quantum attention and feed‑forward modules.
    """
    def __init__(
        self,
        vocab_size: int = 10,
        embed_dim: int = 32,
        num_heads: int = 4,
        num_blocks: int = 2,
        ffn_dim: int = 64,
        num_classes: int = 10,
        n_wires_attention: int = 1,
        n_wires_ffn: int = 4,
        shots: int = 200,
    ) -> None:
        super().__init__()
        # 1. Quantum patch extractor
        self.patch_extractor = QuantumQuanvolutionFilter()
        # 2. Quantum FC layer
        self.quantum_fc = QuantumFCLayer()
        # 3. Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim)
        # 4. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                QuantumTransformerBlock(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    n_wires_attention,
                    n_wires_ffn,
                    shots,
                )
                for _ in range(num_blocks)
            ]
        )
        self.transformer = nn.Sequential(*self.blocks)
        # 5. Classifier
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28)
        # Extract 2×2 patches
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, 0, r:r+2, c:c+2].reshape(x.size(0), -1)
                patches.append(patch)
        patches = torch.stack(patches, dim=1)  # (batch, 14*14, 4)
        # Quantum patch extraction
        patch_feats = self.patch_extractor(patches.view(-1, 4)).view(x.size(0), 14*14, -1)
        # Flatten to sequence
        seq = patch_feats.mean(dim=2, keepdim=True)  # (batch, 14*14, 1)
        seq = seq + self.pos_encoder(seq)
        # Transformer
        out = self.transformer(seq)
        out = out.mean(dim=1)
        return self.classifier(out)

__all__ = ["QuanvolutionHybrid"]
