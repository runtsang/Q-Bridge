import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, transpile, assemble, execute
import torch
import torch.nn as nn

# Quantum self‑attention block that mirrors the classical version
class QuantumSelfAttention:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = qiskit.quantum_info.QuantumRegister(n_qubits, "q")
        self.cr = qiskit.ClassicalRegister(n_qubits, "c")

    def _build(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3*i], i)
            qc.ry(rotation_params[3*i+1], i)
            qc.rz(rotation_params[3*i+2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i+1)
        qc.measure_all()
        return qc

    def run(self, backend, rotation_params, entangle_params, shots: int = 1024):
        qc = self._build(rotation_params, entangle_params)
        job = execute(qc, backend, shots=shots)
        return job.result().get_counts()

# Quantum sampler network that can be used as a hybrid expectation head
class SamplerQNN(nn.Module):
    """
    Quantum sampler with a self‑attention style circuit and a hybrid
    expectation head.  The circuit is parameterised by two vectors:
    rotation_params (size 3*n_qubits) and entangle_params (size n_qubits-1).
    The module can be used as a drop‑in replacement for a classical
    sampler in a hybrid training loop.
    """
    def __init__(self, n_qubits: int = 2, backend=None, shots: int = 1024):
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.attention = QuantumSelfAttention(n_qubits)
        # Trainable parameters
        self.rotation_params = nn.Parameter(torch.randn(3*n_qubits))
        self.entangle_params = nn.Parameter(torch.randn(n_qubits-1))
        self.fc = nn.Linear(n_qubits, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Run quantum circuit to obtain measurement probabilities
        rotation = self.rotation_params.detach().numpy()
        entangle = self.entangle_params.detach().numpy()
        counts = self.attention.run(self.backend, rotation, entangle, shots=self.shots)
        probs = np.array([counts.get(bin(i)[2:].zfill(self.n_qubits), 0) for i in range(2**self.n_qubits)]) / self.shots
        exp = torch.tensor(probs, dtype=torch.float32, device=inputs.device)
        logits = self.fc(exp)
        return torch.softmax(logits, dim=-1)

__all__ = ["SamplerQNN"]
