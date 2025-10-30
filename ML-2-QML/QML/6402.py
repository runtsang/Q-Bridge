import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Qiskit imports for the quantum regression head
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector


# --------------------------------------------------------------------------- #
#  Classical transformer primitives (identical to the ML version)
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.size()
        q = self.q_linear(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.out_linear(out)


class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlockBase(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# --------------------------------------------------------------------------- #
#  Quantum regression head (Qiskit Estimator‑style)
# --------------------------------------------------------------------------- #
class QuantumEstimatorHead(nn.Module):
    """
    A lightweight quantum circuit that maps a scalar input to a scalar output
    via a variational RX rotation.  The circuit is executed on a state‑vector
    simulator; the expectation of the Y Pauli operator is returned.
    """
    def __init__(self):
        super().__init__()
        self.input_param = Parameter("input")
        self.weight_param = Parameter("weight")
        self.circuit = QuantumCircuit(1)
        self.circuit.h(0)
        self.circuit.ry(self.input_param, 0)
        self.circuit.rx(self.weight_param, 0)
        self.obs = SparsePauliOp.from_list([("Y", 1)])
        self.backend = Aer.get_backend("statevector_simulator")
        # Learnable weight shared across the batch
        self.weight = nn.Parameter(torch.randn(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : Tensor of shape (batch, 1)
        Returns a tensor of shape (batch,) containing the expectation value.
        """
        batch = x.shape[0]
        results = []
        for i in range(batch):
            inp = x[i, 0].item()
            bound = self.circuit.bind_parameters(
                {self.input_param: inp, self.weight_param: self.weight.item()}
            )
            qobj = assemble(transpile(bound, self.backend))
            result = self.backend.run(qobj).result()
            sv = Statevector(result.get_statevector())
            exp = sv.expectation_value(self.obs).real
            results.append(exp)
        return torch.tensor(results, device=x.device)


# --------------------------------------------------------------------------- #
#  Hybrid classifier / regressor with quantum head
# --------------------------------------------------------------------------- #
class TextClassifier(nn.Module):
    """
    Transformer‑based model that can optionally use a Qiskit‑based quantum
    regression head.  The ``reg_hidden`` argument triggers the quantum head;
    otherwise a classic linear classifier is used.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int = 2,
        dropout: float = 0.1,
        reg_hidden: Optional[List[int]] = None,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer = nn.Sequential(
            *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
              for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)

        if num_classes == 1 and reg_hidden:
            # Use the quantum Estimator head for regression
            self.head = QuantumEstimatorHead()
        else:
            out_dim = num_classes if num_classes > 2 else 1
            self.head = nn.Linear(embed_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.head(x)
