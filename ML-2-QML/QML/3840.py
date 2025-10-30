import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
from qiskit.circuit.random import random_circuit
import numpy as np
import math

class QiskitQuanvCircuit:
    """Quantum implementation of a 2‑D convolutional filter."""
    def __init__(self, kernel_size: int, backend=None, shots: int = 1024, threshold: float = 0.5):
        self.n_qubits = kernel_size ** 2
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """
        data: 2‑D array of shape (kernel_size, kernel_size) with values in [0,1].
        Returns the average probability of measuring |1> across all qubits.
        """
        data_flat = data.reshape(1, self.n_qubits)
        param_binds = []
        for row in data_flat:
            bind = {theta: np.pi if val > self.threshold else 0 for theta, val in zip(self.theta, row)}
            param_binds.append(bind)

        job = qiskit.execute(self._circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)

class MultiHeadAttentionBase(nn.Module):
    """Shared logic for attention layers."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.size()
        return x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, seq, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch, seq, heads * d_k)

    def forward(self, *args, **kwargs) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Attention that projects queries, keys, values using a small quantum circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 8):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.parameters = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, q_device: tq.QuantumDevice | None = None):
        super().__init__(embed_dim, num_heads, dropout)
        self.q_layer = self.QLayer()
        self.q_device = q_device
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _apply_quantum(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.size()
        outputs = []
        for token in x.unbind(1):
            qdev = self.q_device or tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=1, device=token.device)
            out = self.q_layer(token.unsqueeze(0), qdev)
            outputs.append(out)
        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        k = self._apply_quantum(x)
        q = self._apply_quantum(x)
        v = self._apply_quantum(x)
        k = self._split_heads(k)
        q = self._split_heads(q)
        v = self._split_heads(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = self._merge_heads(out)
        return self.out_proj(out)

class FeedForwardBase(nn.Module):
    """Base for feed‑forward networks."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

class FeedForwardClassical(FeedForwardBase):
    """Two‑layer MLP used after attention."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.fc1 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.fc2 = nn.Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward implemented via a parametric quantum circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.parameters = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim, bias=False)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            out = self.q_layer(token, qdev)
            outputs.append(out)
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

class TransformerBlockBase(nn.Module):
    """Base transformer block."""
    def __init__(self, embed_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

class TransformerBlockQuantum(TransformerBlockBase):
    """Transformer block that uses quantum attention and feed‑forward."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_transformer: int,
        n_qubits_ffn: int,
        dropout: float = 0.1,
        q_device: tq.QuantumDevice | None = None
    ) -> None:
        super().__init__(embed_dim, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, q_device=q_device)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout) if n_qubits_ffn > 0 else FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
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

class HybridTransformerClassifier(nn.Module):
    """
    A transformer‑based classifier that can operate on token sequences or on 2‑D patches
    processed by a lightweight quantum‑friendly convolutional filter.
    """
    def __init__(
        self,
        vocab_size: int | None = None,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_blocks: int = 6,
        ffn_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.1,
        use_conv: bool = False,
        patch_size: int = 2,
        conv_threshold: float = 0.5,
        n_qubits_transformer: int = 0,
        n_qubits_ffn: int = 0,
        q_device: tq.QuantumDevice | None = None,
    ) -> None:
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            if vocab_size is not None:
                raise ValueError("When use_conv=True, vocab_size should be None")
            self.conv_circuit = QiskitQuanvCircuit(patch_size, backend=None, shots=512, threshold=conv_threshold)
            self.token_embed = nn.Linear(1, embed_dim, bias=True)
            self.patch_size = patch_size
        else:
            if vocab_size is None:
                raise ValueError("vocab_size must be provided when use_conv=False")
            self.token_embed = nn.Embedding(vocab_size, embed_dim)

        self.pos_embed = PositionalEncoder(embed_dim)
        if n_qubits_transformer > 0:
            self.transformer = nn.Sequential(
                *[
                    TransformerBlockQuantum(
                        embed_dim,
                        num_heads,
                        ffn_dim,
                        n_qubits_transformer,
                        n_qubits_ffn,
                        dropout,
                        q_device=q_device
                    )
                    for _ in range(num_blocks)
                ]
            )
        else:
            self.transformer = nn.Sequential(
                *[
                    TransformerBlockBase(embed_dim, dropout)
                    for _ in range(num_blocks)
                ]
            )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len) if use_conv=False
           (batch, H, W) if use_conv=True
        """
        if self.use_conv:
            patches = x.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
            batch, num_patches_h, num_patches_w, k, k2 = patches.shape
            num_patches = num_patches_h * num_patches_w
            patches = patches.reshape(batch, num_patches, k, k)
            patch_vals = torch.zeros(batch, num_patches, device=x.device)
            for i in range(batch):
                for j in range(num_patches):
                    patch = patches[i, j]
                    val = self.conv_circuit.run(patch.cpu().numpy())
                    patch_vals[i, j] = val
            tokens = self.token_embed(patch_vals.unsqueeze(-1))
        else:
            tokens = self.token_embed(x)
        tokens = self.pos_embed(tokens)
        out = self.transformer(tokens)
        out = out.mean(dim=1)
        out = self.dropout(out)
        return self.classifier(out)

__all__ = [
    "QiskitQuanvCircuit",
    "MultiHeadAttentionBase",
    "MultiHeadAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardClassical",
    "FeedForwardQuantum",
    "TransformerBlockBase",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "HybridTransformerClassifier",
]
