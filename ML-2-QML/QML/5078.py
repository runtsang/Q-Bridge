"""HybridEstimator: quantum‑augmented estimator with optional quantum attention."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import Parameter
from qiskit.quantum_info import Pauli, Statevector


class QuantumAttention(nn.Module):
    """Quantum self‑attention circuit for 4 qubits."""
    def __init__(self, n_qubits: int = 4):
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = Aer.get_backend("aer_simulator_statevector")

    def forward(
        self,
        x: torch.Tensor,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> torch.Tensor:
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        circuit = QuantumCircuit(qr, cr)

        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)

        for i in range(self.n_qubits - 1):
            circuit.cx(i, i + 1)

        circuit.measure(qr, cr)

        job = execute(circuit, self.backend, shots=1024)
        result = job.result()
        counts = result.get_counts(circuit)

        exp = 0.0
        for bitstring, count in counts.items():
            parity = (-1) ** (bitstring.count("1"))
            exp += parity * count
        exp /= 1024
        return torch.tensor(exp, device=x.device, dtype=x.dtype)


class HybridEstimator(nn.Module):
    """Quantum‑augmented estimator with optional quantum attention."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        use_attention: bool = False,
        use_lstm: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_attention = use_attention
        self.use_lstm = use_lstm

        self.core = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Quantum circuit for estimation
        self.input_param = Parameter("inp")
        self.weight_param = Parameter("wt")
        self.circuit = QuantumCircuit(1)
        self.circuit.h(0)
        self.circuit.ry(self.input_param, 0)
        self.circuit.rx(self.weight_param, 0)
        self.circuit.measure(0, 0)

        # Trainable weight and bias for the quantum layer
        self.weight = nn.Parameter(torch.tensor([0.1]))
        self.bias = nn.Parameter(torch.tensor([0.0]))

        # Scaling and shift buffers
        self.scale = nn.Parameter(torch.tensor([1.0]))
        self.shift = nn.Parameter(torch.tensor([0.0]))

        self.attention = QuantumAttention(n_qubits=4) if use_attention else None
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True) if use_lstm else None

        self.output = nn.Linear(hidden_dim, 1)

        self.backend = Aer.get_backend("aer_simulator_statevector")

    def _quantum_expectation(self, x: torch.Tensor) -> torch.Tensor:
        circ = QuantumCircuit(1)
        circ.h(0)
        circ.ry(x.item(), 0)
        circ.rx(self.weight.item(), 0)
        sv = Statevector.from_instruction(circ)
        y = Pauli("Y")
        exp = (sv.data.conj().T @ y.to_matrix() @ sv.data).real
        return torch.tensor(exp, device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            core_out = self.core(x)
        else:
            batch, seq_len, _ = x.shape
            core_out = self.core(x.view(batch * seq_len, -1))
            core_out = core_out.view(batch, seq_len, -1)

        flat = core_out.flatten()
        q_flat = torch.zeros_like(flat)
        for i, val in enumerate(flat):
            q_flat[i] = self._quantum_expectation(val)

        q_out = q_flat.view_as(core_out)
        q_out = (q_out + self.bias) * self.scale + self.shift

        out = core_out + q_out

        if self.attention is not None:
            rot = np.random.rand(12)  # placeholder
            ent = np.random.rand(3)   # placeholder
            out = self.attention(out, rot, ent)

        if self.lstm is not None:
            out, _ = self.lstm(out)

        out = self.output(out)

        if x.dim() == 2:
            out = out.squeeze(-1)
        return out


__all__ = ["HybridEstimator"]
