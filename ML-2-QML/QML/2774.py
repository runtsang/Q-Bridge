import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
from qiskit.circuit.random import random_circuit

class QuanvCircuit:
    """Quantum convolution (quanvolution) that operates on 2×2 patches."""
    def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
        self.n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data):
        """Run the quantum circuit on classical data.

        Args:
            data: 1D array with length n_qubits.

        Returns:
            float: average probability of measuring |1> across qubits.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim == 2:
            data = data.reshape(-1, self.n_qubits)
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)
        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

class FeedForwardClassical(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class MultiHeadAttentionQuantum(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, q_device: tq.QuantumDevice | None = None):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.d_k)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=False)
        self.qlayer = self._build_qlayer(self.q_device.n_wires)

    def _build_qlayer(self, n_wires: int):
        class QLayer(tq.QuantumModule):
            def __init__(self, n_wires):
                super().__init__()
                self.n_wires = n_wires
                self.encoder = tq.GeneralEncoder(
                    [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
                )
                self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
                self.measure = tq.MeasureAll(tq.PauliZ)
            def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice):
                self.encoder(q_device, x)
                for wire, gate in enumerate(self.params):
                    gate(q_device, wires=wire)
                return self.measure(q_device)
        return QLayer(n_wires)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def _quantum_transform(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (batch, heads, seq, d_k)
        batch, heads, seq, d_k = x.shape
        outputs = []
        for head in range(heads):
            head_tokens = x[:, head, :, :]  # (batch, seq, d_k)
            transformed = []
            for token in head_tokens.unbind(dim=1):  # each token shape (batch, d_k)
                qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
                transformed.append(self.qlayer(token, qdev))
            transformed = torch.stack(transformed, dim=1)  # (batch, seq, d_k)
            outputs.append(transformed)
        outputs = torch.stack(outputs, dim=1)  # (batch, heads, seq, d_k)
        return outputs

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)
        q = self._quantum_transform(q)
        k = self._quantum_transform(k)
        v = self._quantum_transform(v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        out = torch.matmul(scores, v)
        out = out.transpose(1, 2).contiguous().view(x.size(0), -1, self.embed_dim)
        return self.combine_heads(out)

class FeedForwardQuantum(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.qlayer = self._build_qlayer(n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def _build_qlayer(self, n_qubits: int):
        class QLayer(tq.QuantumModule):
            def __init__(self, n_wires):
                super().__init__()
                self.n_wires = n_wires
                self.encoder = tq.GeneralEncoder(
                    [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
                )
                self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
                self.measure = tq.MeasureAll(tq.PauliZ)
                self.q_device = tq.QuantumDevice(n_wires=n_wires)
            def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice):
                self.encoder(q_device, x)
                for wire, gate in enumerate(self.params):
                    gate(q_device, wires=wire)
                return self.measure(q_device)
        return QLayer(n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.size()
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.qlayer.q_device.copy(bsz=token.size(0), device=token.device)
            out = self.qlayer(token, qdev)
            outputs.append(out)
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

class TransformerBlockQuantum(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 n_qubits_transformer: int,
                 n_qubits_ffn: int,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        else:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        return x + self.dropout(ffn_out)

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

class HybridTransformerQuantum(nn.Module):
    """Quantum hybrid transformer that uses quanvolution as front‑end."""
    def __init__(self,
                 image_size: int,
                 kernel_size: int = 2,
                 embed_dim: int = 64,
                 num_heads: int = 8,
                 ffn_dim: int = 256,
                 num_blocks: int = 6,
                 num_classes: int = 10,
                 dropout: float = 0.1,
                 n_qubits_transformer: int = 8,
                 n_qubits_ffn: int = 8,
                 shots: int = 100,
                 threshold: float = 0.5):
        super().__init__()
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.quanv = QuanvCircuit(kernel_size, self.backend, shots, threshold)
        self.seq_len = (image_size - kernel_size + 1) ** 2
        self.embed = nn.Linear(1, embed_dim)
        self.pos_enc = PositionalEncoder(embed_dim)
        self.transformer = nn.Sequential(
            *[TransformerBlockQuantum(embed_dim, num_heads, ffn_dim,
                                      n_qubits_transformer, n_qubits_ffn, dropout)
              for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (batch, 1, H, W)
        batch, _, h, w = x.shape
        patches = []
        for i in range(0, h - self.kernel_size + 1):
            for j in range(0, w - self.kernel_size + 1):
                patch = x[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                patch = patch.view(batch, -1)  # (batch, n_qubits)
                out = []
                for b in range(batch):
                    out.append(self.quanv.run(patch[b].cpu().numpy().reshape(self.kernel_size, self.kernel_size)))
                out = torch.tensor(out, device=x.device).unsqueeze(-1)
                patches.append(out)
        patches = torch.stack(patches, dim=1)  # (batch, seq_len, 1)
        x = self.embed(patches)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

__all__ = [
    "QuanvCircuit",
    "FeedForwardClassical",
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "HybridTransformerQuantum",
]
