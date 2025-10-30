import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
from qiskit.circuit.random import random_circuit

# --------------------------------------------------------------------------- #
#  Quantum‑enhanced attention and feed‑forward
# --------------------------------------------------------------------------- #

class MultiHeadAttentionQuantum(nn.Module):
    """Multi‑head attention where the linear projections are replaced by a
    small variational quantum circuit."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 n_wires: int = 8,
                 q_device: tq.QuantumDevice | None = None) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.n_wires = n_wires
        self.q_device = q_device or tq.QuantumDevice(n_wires=n_wires)
        self.proj = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.combine = nn.Linear(embed_dim, embed_dim)

    def _apply_q(self, vec: torch.Tensor) -> torch.Tensor:
        # vec shape (batch, n_wires)
        qdev = self.q_device.copy(bsz=vec.size(0), device=vec.device)
        self.proj(qdev, vec)
        for gate in self.params:
            gate(qdev)
        return self.measure(qdev)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        # simple linear projection to quantum wires
        proj = x.view(batch, seq_len, self.num_heads, self.d_k)
        q_out = torch.zeros_like(proj)
        for i in range(self.num_heads):
            # collapse batch and seq dims for quantum evaluation
            flat = proj[:, :, i, :].reshape(-1, self.d_k)
            q_res = self._apply_q(flat).reshape(batch, seq_len, self.d_k)
            q_out[:, :, i, :] = q_res
        attn = torch.matmul(q_out, q_out.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, q_out).transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return self.combine(out)


class FeedForwardQuantum(nn.Module):
    """Feed‑forward network implemented by a small variational circuit."""
    def __init__(self,
                 embed_dim: int,
                 ffn_dim: int,
                 n_wires: int = 8,
                 dropout: float = 0.1,
                 q_device: tq.QuantumDevice | None = None) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.n_wires = n_wires
        self.q_device = q_device or tq.QuantumDevice(n_wires=n_wires)
        self.proj = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        )
        self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def _apply_q(self, vec: torch.Tensor) -> torch.Tensor:
        qdev = self.q_device.copy(bsz=vec.size(0), device=vec.device)
        self.proj(qdev, vec)
        for gate in self.params:
            gate(qdev)
        return self.measure(qdev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inter = F.relu(self.linear1(x))
        inter = self.dropout(inter)
        # collapse batch and seq dims for quantum evaluation
        flat = inter.view(-1, inter.size(-1))
        q_res = self._apply_q(flat).view(inter.size())
        return self.linear2(self.dropout(q_res))


# --------------------------------------------------------------------------- #
#  Transformer block and utilities
# --------------------------------------------------------------------------- #

class TransformerBlockQuantum(nn.Module):
    """Transformer block with quantum attention and feed‑forward."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 n_wires_transformer: int = 8,
                 n_wires_ffn: int = 8,
                 dropout: float = 0.1,
                 q_device: tq.QuantumDevice | None = None) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim,
                                               num_heads,
                                               dropout=dropout,
                                               n_wires=n_wires_transformer,
                                               q_device=q_device)
        self.ffn = FeedForwardQuantum(embed_dim,
                                      ffn_dim,
                                      n_wires=n_wires_ffn,
                                      dropout=dropout,
                                      q_device=q_device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        return x + self.dropout(ffn_out)


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, embed_dim, 2) *
                        (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# --------------------------------------------------------------------------- #
#  Quanvolution filter (quantum convolution)
# --------------------------------------------------------------------------- #

class QuanvCircuit:
    """Quantum filter that mimics a 2‑D convolution."""
    def __init__(self,
                 kernel_size: int,
                 backend: qiskit.providers.BaseBackend,
                 shots: int = 1024,
                 threshold: float = 0.5) -> None:
        self.n_qubits = kernel_size ** 2
        self.backend = backend
        self.shots = shots
        self.threshold = threshold
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def run(self, data: torch.Tensor) -> float:
        # data shape (kernel_size, kernel_size)
        flat = data.view(1, self.n_qubits)
        param_binds = []
        for row in flat:
            bind = {self.theta[i]: (math.pi if val > self.threshold else 0) for i, val in enumerate(row)}
            param_binds.append(bind)
        job = qiskit.execute(self.circuit,
                             self.backend,
                             shots=self.shots,
                             parameter_binds=param_binds)
        result = job.result().get_counts(self.circuit)
        count = 0
        for key, val in result.items():
            ones = sum(int(b) for b in key)
            count += ones * val
        return count / (self.shots * self.n_qubits)


# --------------------------------------------------------------------------- #
#  Hybrid transformer with optional quanvolution
# --------------------------------------------------------------------------- #

class HybridTransformerQML(nn.Module):
    """
    Quantum‑enhanced transformer that can use a quanvolution filter as a
    first‑stage embedding.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size for tokenised input. Ignored if ``conv_kernel_size`` is set.
    embed_dim : int
        Dimensionality of hidden embeddings.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer layers.
    ffn_dim : int
        Inner dimension of the feed‑forward network.
    num_classes : int
        Number of output classes.
    dropout : float, optional
        Drop‑out probability.
    conv_kernel_size : int, optional
        If provided, input is treated as a 2‑D image and processed by ``QuanvCircuit``.
    conv_threshold : float, optional
        Threshold for the quanvolution filter.
    n_wires_transformer : int, optional
        Number of wires per attention head.
    n_wires_ffn : int, optional
        Number of wires per feed‑forward quantum block.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 conv_kernel_size: int | None = None,
                 conv_threshold: float = 0.5,
                 n_wires_transformer: int = 8,
                 n_wires_ffn: int = 8) -> None:
        super().__init__()
        self.use_conv = conv_kernel_size is not None
        if self.use_conv:
            self.embedder = QuanvCircuit(kernel_size=conv_kernel_size,
                                         backend=qiskit.Aer.get_backend("qasm_simulator"),
                                         shots=512,
                                         threshold=conv_threshold)
            # output of quanvolution is a scalar; project to embed_dim
            self.proj = nn.Linear(1, embed_dim)
        else:
            self.tokenizer = nn.Embedding(vocab_size, embed_dim)

        self.pos_enc = PositionalEncoder(embed_dim)
        self.layers = nn.ModuleList([
            TransformerBlockQuantum(embed_dim,
                                    num_heads,
                                    ffn_dim,
                                    n_wires_transformer=n_wires_transformer,
                                    n_wires_ffn=n_wires_ffn,
                                    dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_conv:
            # x shape (B, 1, H, W) -> (B, 1)
            x = torch.stack([self.embedder.run(img.squeeze(1)) for img in x], dim=0)
            x = self.proj(x.unsqueeze(1))
        else:
            x = self.tokenizer(x)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "QuanvCircuit",
    "HybridTransformerQML",
]
