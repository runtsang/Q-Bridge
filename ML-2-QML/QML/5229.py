"""Hybrid quantum‑kernel classifier.

This module implements the same interface as the classical version but
uses quantum kernels, quantum transformer blocks, and a quantum hybrid head.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.functional import func_name_dict, op_name_dict
from qiskit import Aer, QuantumCircuit, assemble, transpile
from qiskit.circuit import Parameter

# --------------------------------------------------------------------------- #
# 1. Quantum kernel utilities
# --------------------------------------------------------------------------- #
class QuantumKernalAnsatz(tq.QuantumModule):
    """Encodes classical data into a quantum device and measures overlap."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class QuantumKernel(tq.QuantumModule):
    """Quantum kernel with a small fixed ansatz."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = QuantumKernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry",  "wires": [1]},
                {"input_idx": [2], "func": "ry",  "wires": [2]},
                {"input_idx": [3], "func": "ry",  "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix_quantum(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Return Gram matrix using the quantum kernel."""
    kernel = QuantumKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# 2. Quantum transformer feature extractor
# --------------------------------------------------------------------------- #
class QuantumFFN(tq.QuantumModule):
    """Feed‑forward network realised by a quantum module."""
    def __init__(self, ffn_dim: int, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.parameters = nn.ModuleList(
            [tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(q_device, x)
        for wire, gate in enumerate(self.parameters):
            gate(q_device, wires=wire)
        return self.measure(q_device)

class TransformerFeatureExtractorQuantum(nn.Module):
    """Feature extractor that can be fully quantum or hybrid."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 n_qubits_ffn: int = 0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        if n_qubits_ffn > 0:
            self.ffn = QuantumFFN(ffn_dim, n_qubits_ffn)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.ReLU(),
                nn.Linear(ffn_dim, embed_dim)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# --------------------------------------------------------------------------- #
# 3. Quantum hybrid head
# --------------------------------------------------------------------------- #
class QuantumHybridHead(nn.Module):
    """Hybrid head that uses a parameterised quantum circuit to produce a
    single output probability."""
    def __init__(self, shift: float = math.pi / 2):
        super().__init__()
        self.shift = shift
        self.backend = Aer.get_backend("aer_simulator")
        self.shots = 100
        self.circuit = QuantumCircuit(1)
        self.circuit.h(0)
        self.circuit.barrier()
        self.theta = Parameter("theta")
        self.circuit.ry(self.theta, 0)
        self.circuit.measure_all()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1)
        thetas = x.detach().cpu().numpy().flatten()
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: t} for t in thetas]
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)
        if isinstance(result, list):
            exp = np.array([expectation(item) for item in result])
        else:
            exp = np.array([expectation(result)])
        return torch.tensor(exp, dtype=torch.float32, device=x.device)

# --------------------------------------------------------------------------- #
# 4. Full model
# --------------------------------------------------------------------------- #
class HybridKernelClassifier(nn.Module):
    """End‑to‑end classifier that can be configured as classical or quantum."""
    def __init__(self,
                 num_features: int,
                 kernel_gamma: float = 1.0,
                 transformer_cfg: Tuple[int, int, int] = (64, 4, 256),
                 num_classes: int = 2,
                 use_quantum_kernel: bool = False,
                 transformer_qconfig: Tuple[int, int] = (0, 0)):
        super().__init__()
        # Kernel
        if use_quantum_kernel:
            self.kernel = QuantumKernel()
        else:
            self.kernel = ClassicalKernel(gamma=kernel_gamma)
        # Feature extractor
        n_qffn = transformer_qconfig[1]
        self.feature_extractor = TransformerFeatureExtractorQuantum(
            embed_dim=transformer_cfg[0],
            num_heads=transformer_cfg[1],
            ffn_dim=transformer_cfg[2],
            n_qubits_ffn=n_qffn
        )
        # Hybrid head
        self.head = QuantumHybridHead(shift=math.pi / 2)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, feature_dim)
        features = self.feature_extractor(x)
        logits = self.head(features)
        if self.num_classes == 2:
            return torch.cat((logits, 1 - logits), dim=-1)
        else:
            return logits

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Return Gram matrix using the selected kernel."""
        if isinstance(self.kernel, QuantumKernel):
            return kernel_matrix_quantum(a, b)
        else:
            return kernel_matrix_classical(a, b, gamma=self.kernel.gamma)

__all__ = ["HybridKernelClassifier"]
