"""Hybrid quantum kernel model combining TorchQuantum variational ansatz, Qiskit self‑attention, and fraud‑style scaling."""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from typing import Sequence
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute

class QuantumSelfAttention:
    """Qiskit implementation of a self‑attention style circuit used as a preprocessing block."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(self, rot_params: np.ndarray, ent_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rot_params[3 * i], i)
            circuit.ry(rot_params[3 * i + 1], i)
            circuit.rz(rot_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(ent_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, rot_params: np.ndarray, ent_params: np.ndarray, shots: int = 1024) -> np.ndarray:
        circuit = self._build_circuit(rot_params, ent_params)
        job = execute(circuit, self.backend, shots=shots)
        counts = job.result().get_counts(circuit)
        exp_vals = []
        for q in range(self.n_qubits):
            zero = sum(counts.get('0'*q + '0' + '1'*(self.n_qubits-q-1), 0))
            one = sum(counts.get('0'*q + '1' + '1'*(self.n_qubits-q-1), 0))
            exp_vals.append((zero - one) / shots)
        return np.array(exp_vals)

class QuantumKernelAnsatz(tq.QuantumModule):
    """Variational ansatz that encodes two classical vectors and returns a kernel value."""

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.RandomLayer(n_ops=20, wires=list(range(self.n_wires)))
        self.params = tq.ParameterVector("theta", n_wires * 3)
        self.layer = tq.RY(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        qdev.reset_states(x.shape[0])
        # Encode x
        self.encoder(qdev)
        self.layer(qdev, wires=range(self.n_wires), params=self.params)
        # Uncompute y with inverted parameters
        inv_params = -y[:, None] * self.params
        self.layer(qdev, wires=range(self.n_wires), params=inv_params)

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.forward(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

class HybridKernelModel(tq.QuantumModule):
    """Quantum hybrid kernel that layers a Qiskit self‑attention preprocessing block, a TorchQuantum variational kernel,
    and a fraud‑style scaling factor."""

    def __init__(self, n_qubits: int = 4) -> None:
        super().__init__()
        self.attention = QuantumSelfAttention(n_qubits)
        self.kernel_ansatz = QuantumKernelAnsatz(n_wires=n_qubits)
        self.gamma = 1.0

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Classical‑to‑quantum preprocessing via self‑attention
        rot_params = np.random.uniform(0, 2 * np.pi, 3 * self.attention.n_qubits)
        ent_params = np.random.uniform(0, 2 * np.pi, self.attention.n_qubits - 1)
        att_vals = self.attention.run(rot_params, ent_params)
        # Scale inputs
        x_scaled = x * torch.tensor(att_vals, dtype=x.dtype, device=x.device)
        y_scaled = y * torch.tensor(att_vals, dtype=y.dtype, device=y.device)
        k = self.kernel_ansatz.evaluate(x_scaled, y_scaled)
        return torch.exp(-self.gamma * k)

    @staticmethod
    def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
        model = HybridKernelModel()
        model.gamma = gamma
        return np.array([[model(a_i, b_j).item() for b_j in b] for a_i in a])

__all__ = ["HybridKernelModel"]
