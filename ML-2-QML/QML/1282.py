import pennylane as qml
import pennylane.numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Quantum attention layer
# --------------------------------------------------------------------------- #
class QuantumAttention(nn.Module):
    """
    Variational attention block that uses a small quantum circuit per head.
    The circuit is a depth‑2 rotation‑only circuit implemented in PennyLane.
    """
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 n_qubits: int = 4,
                 circuit_depth: int = 2) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.n_qubits = n_qubits
        self.circuit_depth = circuit_depth

        # Classical linear projections
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Quantum device and circuit
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.q_params = nn.Parameter(np.random.randn(n_qubits, circuit_depth))

        # Final linear projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _quantum_circuit(self, qubits, params):
        """Depth‑2 rotation‑only circuit."""
        for depth in range(self.circuit_depth):
            for i, wire in enumerate(qubits):
                qml.RX(params[wire, depth], wires=wire)
        return qml.expval(qml.PauliZ(qubits[0]))

    def _quantum_forward(self, head: torch.Tensor) -> torch.Tensor:
        # head shape: (batch, seq, d_k)
        batch, seq, dim = head.shape
        # Map each head to a qubit state (truncate if necessary)
        quantum = head[..., :self.n_qubits]
        out = []
        for b in range(batch):
            for s in range(seq):
                # Run the circuit
                @qml.qnode(self.dev, interface="torch")
                def circuit():
                    self._quantum_circuit(range(self.n_qubits), self.q_params)
                    return qml.expval(qml.PauliZ(0))
                out.append(circuit())
        out = torch.stack(out).view(batch, seq, self.n_qubits)
        # Pad back to d_k if needed
        if self.n_qubits < dim:
            pad = torch.zeros(batch, seq, dim - self.n_qubits, device=out.device)
            out = torch.cat([out, pad], dim=-1)
        return out

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Separate heads
        q = q.view(x.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(x.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(x.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        # Classical attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        classical = torch.matmul(attn, v)
        # Quantum augmentation
        quantum = self._quantum_forward(classical)
        mixed = classical + quantum
        # Combine heads
        mixed = mixed.transpose(1, 2).contiguous().view(x.size(0), -1, self.embed_dim)
        return self.out_proj(mixed)

# --------------------------------------------------------------------------- #
# Quantum feed‑forward block
# --------------------------------------------------------------------------- #
class QuantumFeedForward(nn.Module):
    """
    Feed‑forward network realized by a variational quantum circuit.
    The circuit is a depth‑2 rotation‑only circuit applied to each token.
    """
    def __init__(self,
                 embed_dim: int,
                 ffn_dim: int,
                 n_qubits: int = 4,
                 circuit_depth: int = 2) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.n_qubits = n_qubits
        self.circuit_depth = circuit_depth
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.q_params = nn.Parameter(np.random.randn(n_qubits, circuit_depth))

        # Classical linear layers
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def _quantum_circuit(self, params):
        for depth in range(self.circuit_depth):
            for i in range(self.n_qubits):
                qml.RY(params[i, depth], wires=i)
        return qml.expval(qml.PauliZ(0))

    def _quantum_forward(self, token: torch.Tensor) -> torch.Tensor:
        # token shape: (batch, embed_dim)
        batch = token.size(0)
        out = []
        for b in range(batch):
            @qml.qnode(self.dev, interface="torch")
            def circuit():
                self._quantum_circuit(self.q_params)
                return qml.expval(qml.PauliZ(0))
            out.append(circuit())
        return torch.stack(out).view(batch, self.n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical projection
        x = self.linear1(x)
        # Quantum augmentation
        quantum = self._quantum_forward(x)
        # Pad if needed
        if self.n_qubits < self.ffn_dim:
            pad = torch.zeros(x.size(0), self.ffn_dim - self.n_qubits, device=x.device)
            quantum = torch.cat([quantum, pad], dim=-1)
        out = self.linear2(F.relu(quantum))
        return out

# --------------------------------------------------------------------------- #
# Quantum transformer block
# --------------------------------------------------------------------------- #
class QuantumTransformerBlock(nn.Module):
    """
    Transformer block that uses quantum attention and quantum feed‑forward.
    """
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 n_qubits: int = 4,
                 circuit_depth: int = 2) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = QuantumAttention(embed_dim, num_heads, n_qubits, circuit_depth)
        self.ffn = QuantumFeedForward(embed_dim, ffn_dim, n_qubits, circuit_depth)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# --------------------------------------------------------------------------- #
# Quantum text classifier
# --------------------------------------------------------------------------- #
class QuantumTextClassifier(nn.Module):
    """
    Transformer‑based text classifier that uses fully quantum transformer blocks.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 n_qubits: int = 4,
                 circuit_depth: int = 2) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Linear(embed_dim, embed_dim)  # simple learnable pos
        self.blocks = nn.ModuleList([
            QuantumTransformerBlock(embed_dim, num_heads, ffn_dim, n_qubits, circuit_depth)
            for _ in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

__all__ = [
    "QuantumAttention",
    "QuantumFeedForward",
    "QuantumTransformerBlock",
    "QuantumTextClassifier",
]
