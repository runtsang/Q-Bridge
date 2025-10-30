import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchquantum as tq
from torchquantum.functional import func_name_dict, cnot
from typing import List, Optional

# --------------------------------------------------------------------------- #
#  Quantum kernel utilities (adapted from the original seed)
# --------------------------------------------------------------------------- #
class QuantumKernalAnsatz(tq.QuantumModule):
    """Programmable ansatz that encodes two classical vectors into a shared state."""
    def __init__(self, func_list: List[dict]) -> None:
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class QuantumKernel(tq.QuantumModule):
    """Fixed‑parameter quantum kernel."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = QuantumKernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])


class QuantumKernelEmbedding(nn.Module):
    """Encode tokens via a quantum kernel against trainable reference vectors."""
    def __init__(self, embed_dim: int, n_refs: int = 4, n_wires: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_refs = n_refs
        self.n_wires = n_wires
        self.kernel = QuantumKernel()
        # Reference vectors are classical parameters that will be fed into the kernel
        self.refs = nn.Parameter(torch.randn(n_refs, n_wires))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_wires) – token representations.
        Returns:
            (batch, seq_len, n_refs) – kernel‑mapped features.
        """
        batch, seq_len, _ = x.size()
        flat = x.view(-1, self.n_wires)
        # Compute kernel between each token and all reference vectors
        kernels = []
        for ref in self.refs:
            k = self.kernel(flat, ref.unsqueeze(0))
            kernels.append(k)
        # Stack along the reference dimension
        out = torch.stack(kernels, dim=-1).view(batch, seq_len, self.n_refs)
        return out


# --------------------------------------------------------------------------- #
#  Positional encoding
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# --------------------------------------------------------------------------- #
#  Quantum transformer blocks
# --------------------------------------------------------------------------- #
class QuantumMultiHeadAttention(tq.QuantumModule):
    """Attention that maps projections through a small circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 8
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(self.n_wires)
                ]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(self.n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.params):
                gate(q_device, wires=wire)
            for wire in range(self.n_wires - 1):
                cnot(q_device, wires=[wire, wire + 1])
            cnot(q_device, wires=[self.n_wires - 1, 0])
            return self.measure(q_device)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, q_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer()
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.QLayer.n_wires, bsz=1, device="cpu")
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        # Split into heads
        q = x.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = x.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = x.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Apply quantum module to each head separately
        out_heads = []
        for head in range(self.num_heads):
            head_q = q[:, head]                      # (batch, seq_len, d_k)
            head_k = k[:, head]
            head_v = v[:, head]
            qdev = self.q_device.copy(bsz=batch, device=head_q.device)
            # Encode query, key, value through the same circuit
            q_enc = self.q_layer(head_q, qdev)
            k_enc = self.q_layer(head_k, qdev)
            v_enc = self.q_layer(head_v, qdev)
            scores = torch.matmul(q_enc, k_enc.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v_enc)
            out_heads.append(out.unsqueeze(1))
        out = torch.cat(out_heads, dim=1)           # (batch, num_heads, seq_len, d_k)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return self.out_proj(out)


class QuantumFeedForward(tq.QuantumModule):
    """Feed‑forward realized by a small quantum circuit followed by classical layers."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int) -> None:
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.params = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.params):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim, bias=False)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        out = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            out.append(self.q_layer(token, qdev))
        out = torch.stack(out, dim=1)               # (batch, seq_len, n_qubits)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class QuantumTransformerBlock(tq.QuantumModule):
    """A single transformer block that can mix classical and quantum sub‑modules."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 n_qubits_attn: int, n_qubits_ffn: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = QuantumMultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = QuantumFeedForward(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
#  Unified classifier API (quantum version)
# --------------------------------------------------------------------------- #
class QTransformerClassifier(nn.Module):
    """
    Quantum‑first transformer classifier.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    embed_dim : int
        Token‑embedding dimension.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Feed‑forward hidden size.
    num_classes : int
        Output classes.
    dropout : float, default 0.1
        Drop‑out probability.
    use_quantum_blocks : bool, default True
        If True, use the quantum transformer blocks; otherwise fall back to classical ones.
    use_kernel : bool, default False
        If True, prepend a quantum kernel embedding.
    n_qubits_attn : int, default 8
        Number of qubits used in the attention circuit.
    n_qubits_ffn : int, default 4
        Number of qubits used in the feed‑forward circuit.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_quantum_blocks: bool = True,
        use_kernel: bool = False,
        n_qubits_attn: int = 8,
        n_qubits_ffn: int = 4,
    ) -> None:
        super().__init__()
        self.use_quantum_blocks = use_quantum_blocks
        self.use_kernel = use_kernel

        if use_kernel:
            self.token_embedding = QuantumKernelEmbedding(embed_dim, n_refs=embed_dim, n_wires=n_qubits_attn)
        else:
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        self.pos_encoder = PositionalEncoder(embed_dim)
        if use_quantum_blocks:
            self.blocks = nn.ModuleList(
                [QuantumTransformerBlock(embed_dim, num_heads, ffn_dim,
                                         n_qubits_attn, n_qubits_ffn, dropout)
                 for _ in range(num_blocks)]
            )
        else:
            # Fallback to classical blocks for compatibility
            self.blocks = nn.ModuleList(
                [TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
                 for _ in range(num_blocks)]
            )

        self.dropout = nn.Dropout(dropout)
        out_dim = 1 if num_classes <= 2 else num_classes
        self.classifier = nn.Linear(embed_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.LongTensor
            Input token indices of shape (batch, seq_len).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, num_classes) or (batch, 1).
        """
        if self.use_kernel:
            emb = self.token_embedding(x)
        else:
            emb = self.token_embedding(x)

        emb = self.pos_encoder(emb)

        for block in self.blocks:
            emb = block(emb)

        pooled = emb.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


__all__ = ["QTransformerClassifier"]
