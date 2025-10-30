import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit.random import random_circuit

# ------------------------------------------------------------------
# 1. Quantum self‑attention using Qiskit
# ------------------------------------------------------------------
class QuantumSelfAttention:
    """
    Quantum circuit that implements a self‑attention style block.
    Each token is encoded into a qubit; rotations and controlled‑RX gates
    realise the query‑key‑value interaction.
    """
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qreg = QuantumRegister(n_qubits, 'q')
        self.creg = ClassicalRegister(n_qubits, 'c')
        self.backend = Aer.get_backend('qasm_simulator')
        self.shots = 1024

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circ = QuantumCircuit(self.qreg, self.creg)
        for i in range(self.n_qubits):
            circ.rx(rotation_params[3 * i], self.qreg[i])
            circ.ry(rotation_params[3 * i + 1], self.qreg[i])
            circ.rz(rotation_params[3 * i + 2], self.qreg[i])
        for i in range(self.n_qubits - 1):
            circ.crx(entangle_params[i], self.qreg[i], self.qreg[i + 1])
        circ.measure(self.qreg, self.creg)
        return circ

    def run(self, x: torch.Tensor, rotation_params: np.ndarray, entangle_params: np.ndarray) -> torch.Tensor:
        """
        x: (batch, seq, embed_dim) – we flatten the last dim to match the number of qubits.
        Returns a probability‑based attention map.
        """
        batch, seq, _ = x.size()
        # Simplify: only use the first n_qubits dimensions of each token
        x_flat = x[:, :, :self.n_qubits].cpu().numpy()
        probs = np.zeros((batch, seq, seq))
        for b in range(batch):
            for i in range(seq):
                for j in range(seq):
                    circ = self._build_circuit(rotation_params, entangle_params)
                    # Bind qubit states to the input token
                    param_binds = {f'theta{i}': np.pi if x_flat[b, i, j] > 0.5 else 0 for i in range(self.n_qubits)}
                    job = execute(circ, self.backend, shots=self.shots, parameter_binds=[param_binds])
                    counts = job.result().get_counts(circ)
                    # Convert counts to probability of measuring |1> on the first qubit
                    prob = sum(int(bit) for key, val in counts.items() for bit in key) / (self.shots * self.n_qubits)
                    probs[b, i, j] = prob
        return torch.tensor(probs, device=x.device, dtype=torch.float32)

# ------------------------------------------------------------------
# 2. Quantum feed‑forward using a random circuit
# ------------------------------------------------------------------
class QuantumFeedForward:
    """
    A small quantum feed‑forward network based on a random circuit
    followed by parameterised RX, RY, RZ gates.
    """
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qreg = QuantumRegister(n_qubits, 'q')
        self.creg = ClassicalRegister(n_qubits, 'c')
        self.backend = Aer.get_backend('qasm_simulator')
        self.shots = 1024
        self.circuit = QuantumCircuit(self.qreg, self.creg)
        # Random layer
        self.circuit += random_circuit(n_qubits, 2)
        # Parameterised gates
        for i in range(n_qubits):
            self.circuit.rx(0, self.qreg[i])
        self.circuit.measure_all()

    def run(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq, embed_dim) – flatten to n_qubits per token.
        Returns a tensor of shape (batch, seq, embed_dim) after quantum processing.
        """
        batch, seq, _ = x.size()
        out = np.zeros((batch, seq, self.n_qubits))
        for b in range(batch):
            for s in range(seq):
                param_binds = {f'theta{i}': np.pi if x[b, s, i] > 0.5 else 0 for i in range(self.n_qubits)}
                job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=[param_binds])
                counts = job.result().get_counts(self.circuit)
                # Convert counts to expectation values
                exp = sum((int(bit) - 0.5) * val for key, val in counts.items() for bit in key)
                out[b, s, :] = exp / self.shots
        return torch.tensor(out, device=x.device, dtype=torch.float32)

# ------------------------------------------------------------------
# 3. Hybrid transformer primitives
# ------------------------------------------------------------------
class TransformerBlockBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerBlockQuantum(TransformerBlockBase):
    """
    Hybrid block that can mix quantum attention and/or quantum feed‑forward.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        use_q_attn: bool = False,
        use_q_ffn: bool = False,
        n_qubits: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.use_q_attn = use_q_attn
        self.use_q_ffn = use_q_ffn
        self.n_qubits = n_qubits
        if use_q_attn:
            self.attn = QuantumSelfAttention(n_qubits)
        else:
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        if use_q_ffn:
            self.ffn = QuantumFeedForward(n_qubits)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, embed_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantum attention expects rotation & entangle params – we use simple random ones
        if self.use_q_attn:
            rot_params = np.random.uniform(0, 2 * np.pi, size=(self.n_qubits * 3,))
            ent_params = np.random.uniform(0, 2 * np.pi, size=(self.n_qubits - 1,))
            attn_out = self.attn.run(x, rot_params, ent_params)
        else:
            attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        if self.use_q_ffn:
            ffn_out = self.ffn.run(x)
        else:
            ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# ------------------------------------------------------------------
# 4. Positional encoding
# ------------------------------------------------------------------
class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

# ------------------------------------------------------------------
# 5. Quantum‑convolutional preprocessor (Qiskit)
# ------------------------------------------------------------------
class QuantumConvPreprocessor(nn.Module):
    """
    Simple quantum filter based on a random circuit measuring each qubit.
    The output is a single scalar per patch, used as a token embedding.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.5, shots: int = 512) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.backend = Aer.get_backend('qasm_simulator')
        self.circuit = QuantumCircuit(kernel_size ** 2, kernel_size ** 2, kernel_size ** 2)
        # Random circuit
        self.circuit += random_circuit(kernel_size ** 2, 2)
        self.circuit.measure_all()

    def run(self, patch: np.ndarray) -> float:
        # patch shape: (kernel, kernel)
        data = patch.flatten()
        param_binds = {f'theta{i}': np.pi if data[i] > self.threshold else 0 for i in range(data.size)}
        job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=[param_binds])
        counts = job.result().get_counts(self.circuit)
        # average probability of measuring |1> across all qubits
        total = 0
        for key, val in counts.items():
            ones = sum(int(b) for b in key)
            total += ones * val
        return total / (self.shots * self.circuit.num_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, H, W = x.size()
        tokens = []
        for i in range(0, H - self.kernel_size + 1, self.kernel_size):
            for j in range(0, W - self.kernel_size + 1, self.kernel_size):
                patch = x[:, :, i:i + self.kernel_size, j:j + self.kernel_size].cpu().numpy()
                val = self.run(patch[0])  # use first sample for simplicity
                tokens.append(torch.tensor(val, device=x.device, dtype=torch.float32))
        if not tokens:
            return torch.zeros(batch, 0, dtype=torch.float32, device=x.device)
        return torch.stack(tokens, dim=1).unsqueeze(-1)  # shape: (batch, seq, 1)

# ------------------------------------------------------------------
# 6. Hybrid text classifier (Qiskit version)
# ------------------------------------------------------------------
class HybridTextClassifier(nn.Module):
    """
    Identical API to the TorchQuantum version but implemented with Qiskit circuits.
    Supports classical, quantum, or hybrid transformer blocks and an optional
    quantum convolutional preprocessor for image‑like inputs.
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
        use_q_attn: bool = False,
        use_q_ffn: bool = False,
        n_qubits: int = 4,
        use_qconv: bool = False,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.use_qconv = use_qconv
        if use_qconv:
            self.preprocessor = QuantumConvPreprocessor()
        else:
            self.preprocessor = None

        blocks = []
        for _ in range(num_blocks):
            if use_quantum:
                blocks.append(
                    TransformerBlockQuantum(
                        embed_dim,
                        num_heads,
                        ffn_dim,
                        use_q_attn=use_q_attn,
                        use_q_ffn=use_q_ffn,
                        n_qubits=n_qubits,
                        dropout=dropout,
                    )
                )
            else:
                blocks.append(
                    TransformerBlockClassical(
                        embed_dim, num_heads, ffn_dim, dropout
                    )
                )
        self.transformer = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_qconv:
            tokens = self.preprocessor(x)  # shape: (batch, seq, 1)
            x = tokens
        else:
            tokens = self.token_embedding(x)  # shape: (batch, seq, embed_dim)
            x = tokens
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

__all__ = [
    "QuantumSelfAttention",
    "QuantumFeedForward",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "QuantumConvPreprocessor",
    "HybridTextClassifier",
]
