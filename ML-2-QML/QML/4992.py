from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import Parameter
from qiskit.circuit.random import random_circuit
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict

class QuanvFilter:
    """Quantum analogue of a 2‑D convolutional filter used in quanvolution layers."""
    def __init__(self, kernel_size: int, backend, shots: int, threshold: int):
        self.n_qubits = kernel_size ** 2
        self.circuit = QuantumCircuit(self.n_qubits)
        self.parameters = [Parameter(f"θ{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.parameters[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, depth=2)
        self.circuit.measure_all()
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def evaluate(self, patch: np.ndarray) -> float:
        """Return the average probability of measuring |1> over all qubits."""
        flat = patch.reshape(1, self.n_qubits)
        binds = [{p: np.pi if val > self.threshold else 0 for p, val in zip(self.parameters, row)}
                 for row in flat]
        job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=binds)
        result = job.result().get_counts(self.circuit)
        total = self.shots * self.n_qubits
        return sum(sum(int(bit) for bit in key) * cnt for key, cnt in result.items()) / total

class QuantumAttention:
    """Variational attention circuit that mimics a scaled‑dot‑product block."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qreg = QuantumRegister(n_qubits)
        self.creg = ClassicalRegister(n_qubits)
        self.base_circuit = QuantumCircuit(self.qreg, self.creg)

    def _assemble(self, rot: np.ndarray, ent: np.ndarray) -> QuantumCircuit:
        circ = QuantumCircuit(self.qreg, self.creg)
        for i in range(self.n_qubits):
            circ.rx(rot[3 * i], i)
            circ.ry(rot[3 * i + 1], i)
            circ.rz(rot[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circ.crx(ent[i], i, i + 1)
        circ.measure_all()
        return circ

    def execute(self, backend, rot: np.ndarray, ent: np.ndarray, shots=1024) -> dict:
        circ = self._assemble(rot, ent)
        job = execute(circ, backend, shots=shots)
        return job.result().get_counts(circ)

class QuantumKernel(tq.QuantumModule):
    """Fixed ansatz that computes an overlap between two classical vectors."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.qdev = tq.QuantumDevice(n_wires=self.n_wires)
        self.ry = tq.RY(has_params=True, trainable=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, self.n_wires)
        y = y.reshape(-1, self.n_wires)
        self.qdev.reset_states(x.shape[0])
        for w in range(self.n_wires):
            self.ry(self.qdev, wires=w, params=x[:, w])
        for w in range(self.n_wires):
            self.ry(self.qdev, wires=w, params=-y[:, w])
        return torch.abs(self.qdev.states.view(-1)[0])

class QuantumFC(tq.QuantumModule):
    """Tiny quantum fully‑connected head that maps the attention outcome to 4 features."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.fc = nn.Linear(self.n_wires, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

class SelfAttentionHybrid(tq.QuantumModule):
    """
    Quantum‑centric self‑attention model that stitches together a quanvolution filter,
    a fixed quantum kernel, a variational attention circuit, and a quantum FC head.
    """
    def __init__(self,
                 n_qubits: int = 4,
                 conv_size: int = 2,
                 conv_threshold: int = 127):
        super().__init__()
        self.attention = QuantumAttention(n_qubits)
        self.quanv = QuanvFilter(conv_size,
                                 backend=Aer.get_backend("qasm_simulator"),
                                 shots=100,
                                 threshold=conv_threshold)
        self.kernel = QuantumKernel()
        self.fc_head = QuantumFC()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, embed_dim)
        # 1) Quanvolution step
        quanv_vals = torch.tensor([self.quanv.evaluate(p.cpu().numpy())
                                   for p in x], dtype=torch.float32, device=x.device)
        # 2) Kernel encoding (here we just reuse the original data)
        kernel_vals = self.kernel(x.reshape(-1, 4), x.reshape(-1, 4))
        # 3) Variational attention with random parameters
        rot_params = np.random.rand(self.attention.n_qubits * 3)
        ent_params = np.random.rand(self.attention.n_qubits - 1)
        counts = self.attention.execute(Aer.get_backend("qasm_simulator"),
                                        rot_params, ent_params, shots=1024)
        # Convert counts into a probability vector per qubit
        total_shots = sum(counts.values())
        prob_vec = torch.zeros(self.attention.n_qubits, device=x.device)
        for outcome, freq in counts.items():
            for i, bit in enumerate(outcome):
                if bit == '1':
                    prob_vec[i] += freq
        prob_vec /= total_shots
        # 4) Final projection
        return self.fc_head(prob_vec.unsqueeze(0))

__all__ = ["SelfAttentionHybrid"]
