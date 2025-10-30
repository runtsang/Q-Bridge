import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding used in transformers."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) *
                             (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class GatedMultiHeadAttention(nn.Module):
    """Multi‑head attention with a learnable gate per head."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.ones(num_heads))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, E = x.size()
        q = self.q_linear(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        gate = torch.sigmoid(self.gate).view(1, self.num_heads, 1, 1)
        out = out * gate
        out = out.transpose(1, 2).contiguous().view(B, T, E)
        return self.out_linear(out)

class FeedForward(nn.Module):
    """Two‑layer feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class QuantumFeedForward(nn.Module):
    """Feed‑forward network realised by a variational quantum circuit."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, n_layers: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_linear = nn.Linear(embed_dim, n_qubits)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.q_device = qml.device("default.qubit", wires=n_qubits)
        self.params = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self._qnode = qml.qnode(self._quantum_circuit, device=self.q_device, interface="torch", diff_method="parameter-shift")

    def _quantum_circuit(self, x: torch.Tensor, params: torch.Tensor) -> list[torch.Tensor]:
        for l in range(self.n_layers):
            for q in range(self.n_qubits):
                qml.RX(params[l, q, 0], wires=q)
                qml.RY(params[l, q, 1], wires=q)
                qml.RZ(params[l, q, 2], wires=q)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[self.n_qubits - 1, 0])
        return [qml.expval(qml.PauliZ(q)) for q in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, E = x.size()
        x = x.reshape(-1, E)
        x_qubits = self.input_linear(x)
        outputs = []
        for i in range(x_qubits.size(0)):
            out = self._qnode(x_qubits[i], self.params)
            outputs.append(out)
        out = torch.stack(outputs, dim=0)
        out = self.dropout(out)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        return out.reshape(B, T, E)

class TransformerBlock(nn.Module):
    """Standard transformer block with gated attention."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = GatedMultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class TransformerBlockQuantum(nn.Module):
    """Transformer block that uses a quantum feed‑forward sub‑module."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, n_qubits: int, n_layers: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = GatedMultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = QuantumFeedForward(embed_dim, ffn_dim, n_qubits, n_layers, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class HybridTransformer(nn.Module):
    """Transformer‑based text classifier with optional quantum feed‑forward."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_quantum: bool = False,
        n_qubits: int = 0,
        n_layers: int = 0,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional = PositionalEncoding(embed_dim)
        if use_quantum and n_qubits > 0:
            self.blocks = nn.ModuleList(
                [
                    TransformerBlockQuantum(
                        embed_dim, num_heads, ffn_dim, n_qubits, n_layers, dropout
                    )
                    for _ in range(num_blocks)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
                    for _ in range(num_blocks)
                ]
            )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.positional(tokens)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)
