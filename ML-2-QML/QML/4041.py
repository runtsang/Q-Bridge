"""UnifiedEstimatorQNN: quantum‑centric implementation.

The module implements the same API as the classical version but replaces the
kernel and transformer blocks with variational quantum circuits.  The
``mode`` parameter selects between a classical feed‑forward regressor,
a quantum‑kernel ridge regressor, or a quantum‑transformer classifier.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import torchquantum as tq
from torchquantum.functional import func_name_dict

# --------------------------------------------------------------------------- #
# 1. Classical fallback (identical to the classical implementation)
# --------------------------------------------------------------------------- #
class EstimatorNN(nn.Module):
    """Shallow fully‑connected regression network (classical fallback)."""
    def __init__(self, input_dim: int = 2, hidden_sizes: List[int] = None) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [8, 4]
        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.Tanh())
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# --------------------------------------------------------------------------- #
# 2. Quantum kernel ansatz
# --------------------------------------------------------------------------- #
class QuantumKernelAnsatz(tq.QuantumModule):
    """Encodes two samples and performs a reverse encoding for the inner product."""
    def __init__(self, gates: List[dict]) -> None:
        super().__init__()
        self.gates = gates

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice,
                x: torch.Tensor,
                y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        # encode x
        for g in self.gates:
            params = x[:, g["input_idx"]]
            func_name_dict[g["func"]](q_device, wires=g["wires"], params=params)
        # reverse encode y with negative angles
        for g in reversed(self.gates):
            params = -y[:, g["input_idx"]]
            func_name_dict[g["func"]](q_device, wires=g["wires"], params=params)

class QuantumKernel(tq.QuantumModule):
    """Quantum kernel that returns the absolute overlap."""
    def __init__(self, gates: List[dict] = None) -> None:
        super().__init__()
        if gates is None:
            gates = [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        self.ansatz = QuantumKernelAnsatz(gates)
        self.q_device = tq.QuantumDevice(n_wires=len(gates))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.ansatz(self.q_device, x, y)
        # overlap is the absolute value of the first state amplitude
        return torch.abs(self.q_device.states.view(-1)[0])

def quantum_kernel_matrix(a: List[torch.Tensor], b: List[torch.Tensor]) -> np.ndarray:
    kernel = QuantumKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

class QuantumKernelRegressor(nn.Module):
    """Kernel ridge regression built on the quantum kernel."""
    def __init__(self, lambda_reg: float = 1e-3) -> None:
        super().__init__()
        self.lambda_reg = lambda_reg
        self.train_X = None
        self.train_y = None
        self.alpha = None

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.train_X, self.train_y = X, y
        K = torch.tensor(quantum_kernel_matrix(X, X))
        K += self.lambda_reg * torch.eye(K.size(0))
        self.alpha = torch.linalg.solve(K, y)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.alpha is None:
            raise RuntimeError("Model not fitted")
        K_test = torch.tensor(quantum_kernel_matrix(X, self.train_X))
        return K_test @ self.alpha

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward(X)

# --------------------------------------------------------------------------- #
# 3. Quantum transformer components
# --------------------------------------------------------------------------- #
class QuantumMultiHeadAttention(tq.QuantumModule):
    """Quantum mapping of attention projections."""
    def __init__(self, embed_dim: int, num_heads: int, n_qubits: int = 8) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(n_qubits)
            ]
        )
        self.parameters = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_qubits)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, token: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(q_device, token)
        for gate in self.parameters:
            gate(q_device)
        # simple ring entanglement
        for i in range(self.n_qubits - 1):
            tq.cnot(q_device, wires=[i, i + 1])
        tq.cnot(q_device, wires=[self.n_qubits - 1, 0])
        return self.measure(q_device)

class QuantumFeedForward(tq.QuantumModule):
    """Quantum feed‑forward block."""
    def __init__(self, n_qubits: int) -> None:
        super().__init__()
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.parameters = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, token: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(q_device, token)
        for gate in self.parameters:
            gate(q_device)
        return self.measure(q_device)

class QuantumTransformerBlock(nn.Module):
    """Transformer block that uses quantum modules for attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 n_qubits_attn: int = 8, n_qubits_ffn: int = 8,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = QuantumMultiHeadAttention(embed_dim, num_heads, n_qubits_attn)
        self.ffn = QuantumFeedForward(n_qubits_ffn)
        self.linear1 = nn.Linear(n_qubits_ffn, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # classical attention projection (identity for simplicity)
        proj = x
        q_device = tq.QuantumDevice(n_wires=self.attn.n_qubits, bsz=proj.size(0))
        attn_out = torch.stack([self.attn(tok, q_device) for tok in proj.unbind(dim=1)], dim=1)
        x = self.norm1(x + self.dropout(attn_out))
        # quantum feed‑forward
        ffn_out = torch.stack([self.ffn(tok, q_device) for tok in x.unbind(dim=1)], dim=1)
        ffn_out = self.linear1(self.dropout(ffn_out))
        ffn_out = self.linear2(F.relu(ffn_out))
        return self.norm2(x + self.dropout(ffn_out))

class QuantumPositionalEncoder(nn.Module):
    """Sinusoidal positional encoding (identical to classical)."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class QuantumTextClassifier(nn.Module):
    """Transformer‑based text classifier with quantum submodules."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 n_qubits_attn: int = 8,
                 n_qubits_ffn: int = 8) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = QuantumPositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[QuantumTransformerBlock(embed_dim, num_heads, ffn_dim,
                                      n_qubits_attn, n_qubits_ffn, dropout)
              for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

# --------------------------------------------------------------------------- #
# 4. Unified estimator
# --------------------------------------------------------------------------- #
class UnifiedEstimatorQNN(nn.Module):
    """Hybrid estimator with interchangeable backends, quantum‑centric implementation.

    Parameters
    ----------
    mode : {'classical', 'kernel', 'transformer'}
        Backend to use.
    kwargs : dict
        Additional arguments forwarded to the chosen backend.
    """
    def __init__(self, mode: str = "kernel", **kwargs) -> None:
        super().__init__()
        self.mode = mode
        if mode == "classical":
            self.backend = EstimatorNN(**kwargs)
        elif mode == "kernel":
            self.backend = QuantumKernelRegressor(**kwargs)
        elif mode == "transformer":
            self.backend = QuantumTextClassifier(**kwargs)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backend(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        if hasattr(self.backend, "fit"):
            self.backend.fit(X, y)
        else:
            raise AttributeError("Backend does not implement fit")

__all__ = ["UnifiedEstimatorQNN", "EstimatorNN", "QuantumKernelRegressor", "QuantumTextClassifier"]
