import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer

def SelfAttention():
    """
    Quantum self‑attention built using Qiskit.
    Returns a class with a run method that executes a small variational circuit.
    """
    class QuantumSelfAttention:
        def __init__(self, n_qubits: int):
            self.n_qubits = n_qubits
            self.qr = QuantumRegister(n_qubits)
            self.cr = ClassicalRegister(n_qubits)

        def _build_circuit(self, rotation_params, entangle_params):
            qc = QuantumCircuit(self.qr, self.cr)
            for i in range(self.n_qubits):
                qc.rx(rotation_params[3 * i], i)
                qc.ry(rotation_params[3 * i + 1], i)
                qc.rz(rotation_params[3 * i + 2], i)
            for i in range(self.n_qubits - 1):
                qc.crx(entangle_params[i], i, i + 1)
            qc.measure(self.qr, self.cr)
            return qc

        def run(self, rotation_params, entangle_params, shots=1024):
            qc = self._build_circuit(rotation_params, entangle_params)
            backend = Aer.get_backend('qasm_simulator')
            job = execute(qc, backend, shots=shots)
            return job.result().get_counts(qc)

    return QuantumSelfAttention(n_qubits=4)

class PositionalEncoder(nn.Module):
    """
    Sinusoidal positional encoding as used in the reference transformer.
    """
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32) *
            (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

class TransformerBlock(nn.Module):
    """
    Classical transformer block used when the quantum flag is off.
    """
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class QuantumProjection(tq.QuantumModule):
    """
    Simple quantum layer that encodes a tensor into a circuit, applies parameterized gates,
    entangles and measures.  It is used inside the quantum attention and feed‑forward blocks.
    """
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(q_device, x)
        for wire, gate in enumerate(self.params):
            gate(q_device, wires=wire)
        for wire in range(self.n_wires - 1):
            tqf.cnot(q_device, wires=[wire, wire + 1])
        return self.measure(q_device)

class QuantumAttention(tq.QuantumModule):
    """
    Attention that applies a quantum projection to each head and then performs the usual
    attention computation on the resulting measurement vectors.
    """
    def __init__(self, embed_dim: int, num_heads: int, n_wires: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.projections = nn.ModuleList([QuantumProjection(n_wires) for _ in range(num_heads)])

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        proj = []
        for head in range(self.num_heads):
            head_input = x.view(batch, seq_len, self.d_k)
            qdev = q_device.copy(bsz=batch * seq_len, device=head_input.device)
            out = self.projections[head](head_input.reshape(-1, self.d_k), qdev)
            proj.append(out.reshape(batch, seq_len, self.d_k))
        return torch.stack(proj, dim=1)

class QuantumFeedForward(tq.QuantumModule):
    """
    Feed‑forward network realised by a quantum module followed by classical linear layers.
    """
    def __init__(self, embed_dim: int, ffn_dim: int, n_wires: int):
        super().__init__()
        self.quantum = QuantumProjection(n_wires)
        self.fc1 = nn.Linear(n_wires, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        flat = x.view(-1, x.size(-1))
        qdev = q_device.copy(bsz=flat.size(0), device=flat.device)
        out = self.quantum(flat, qdev)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out.view(batch, seq_len, -1)

class QuantumTransformerBlock(tq.QuantumModule):
    """
    A transformer block that contains a quantum attention and a quantum feed‑forward sub‑module.
    """
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, n_wires_attn: int, n_wires_ffn: int):
        super().__init__()
        self.attn = QuantumAttention(embed_dim, num_heads, n_wires_attn)
        self.ffn = QuantumFeedForward(embed_dim, ffn_dim, n_wires_ffn)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        attn_out = self.attn(x, q_device)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x, q_device)
        return self.norm2(x + self.dropout(ffn_out))

class HybridSelfAttentionTransformer(nn.Module):
    """
    Quantum‑enhanced transformer that falls back to the classical implementation
    when the quantum flag is False.  The class keeps the same public API as the
    classical version so it can be used as a drop‑in replacement.
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
        use_quantum: bool = False,
        n_wires_attn: int = 8,
        n_wires_ffn: int = 8,
    ):
        super().__init__()
        self.use_quantum = use_quantum
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        if use_quantum:
            self.blocks = nn.ModuleList([
                QuantumTransformerBlock(embed_dim, num_heads, ffn_dim, n_wires_attn, n_wires_ffn)
                for _ in range(num_blocks)
            ])
        else:
            self.blocks = nn.ModuleList([
                TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
        self.q_device = tq.QuantumDevice(n_wires=max(n_wires_attn, n_wires_ffn))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)
        for block in self.blocks:
            if self.use_quantum:
                x = block(x, self.q_device)
            else:
                x = block(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)
