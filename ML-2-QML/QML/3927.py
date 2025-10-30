"""QML implementation of a hybrid convolutional network with quantum self‑attention
and a variational quantum expectation head.  The design mirrors the classical
baseline but replaces all classical primitives with quantum circuits.

Shared class name: HybridAttentionQCNet
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, assemble
from qiskit.providers.aer import AerSimulator

class QuantumCircuitWrapper:
    """Two‑qubit variational circuit used as a differentiable expectation head."""
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self.n_qubits = n_qubits
        self.backend  = backend
        self.shots    = shots
        self.circuit = QuantumCircuit(n_qubits, name="hybrid")
        self.theta = qiskit.circuit.Parameter("theta")

        # Simple entangling layer
        for q in range(n_qubits):
            self.circuit.h(q)
        self.circuit.barrier()
        self.circuit.ry(self.theta, list(range(n_qubits)))
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for a batch of angles."""
        compiled = transpile(self.circuit, self.backend)
        param_binds = [{self.theta: theta} for theta in thetas]
        qobj = assemble(compiled, shots=self.shots, parameter_binds=param_binds)
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        return np.array([self._expectation(result)])

    @staticmethod
    def _expectation(counts: dict) -> float:
        counts_arr = np.array(list(counts.values()))
        probs = counts_arr / sum(counts_arr)
        states = np.array([int(k, 2) for k in counts.keys()], dtype=float)
        return float(np.sum(states * probs))

class QuantumSelfAttention:
    """Quantum self‑attention block implemented with a 4‑qubit circuit."""
    def __init__(self, n_qubits: int = 4, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend  = backend or AerSimulator()
        self.shots    = shots
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr, name="attention")
        # Rotate each qubit
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        # Entangle neighboring qubits
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = 1024) -> dict:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, self.backend, shots=shots)
        return job.result().get_counts(circuit)

    def expectation_vector(self, counts: dict) -> np.ndarray:
        """Return a vector of Pauli‑Z expectations for each qubit."""
        probs = np.array(list(counts.values()), dtype=float)
        probs /= probs.sum()
        states = np.array([int(k, 2) for k in counts.keys()], dtype=float)
        expectation = np.zeros(self.n_qubits)
        for i in range(self.n_qubits):
            bit_mask = 1 << i
            bit_vals = ((states & bit_mask) >> i).astype(float)
            exp_z = np.sum((1 - 2 * bit_vals) * probs)
            expectation[i] = exp_z
        return expectation

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a variational quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Expectation values are computed on CPU; convert to torch tensor
        expectation = self.quantum_circuit.run(inputs.tolist())
        return torch.tensor(expectation, dtype=torch.float32)

class HybridAttentionQCNet(nn.Module):
    """CNN + quantum self‑attention + variational quantum head."""
    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1   = nn.Linear(55815, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, embed_dim)
        self.attention = QuantumSelfAttention(n_qubits=embed_dim, shots=1024)
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(embed_dim, backend, shots=100, shift=np.pi / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv backbone
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))

        # Quantum self‑attention on the last hidden dimension
        x = self.fc3(x)                                 # (batch, embed_dim)
        batch_size = x.size(0)
        # Generate rotation and entangle parameters from the features
        rot = np.tile(x.detach().cpu().numpy(), 3).flatten()   # 3 * embed_dim
        ent = np.zeros(self.attention.n_qubits - 1)             # simple entangle params
        counts = self.attention.run(rot, ent, shots=self.attention.shots)
        attn_vec = self.attention.expectation_vector(counts)   # (embed_dim,)
        attn_tensor = torch.tensor(attn_vec, dtype=torch.float32, device=x.device)

        # Hybrid quantum head
        probs = self.hybrid(attn_tensor)                    # (batch, 1)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["QuantumCircuitWrapper", "QuantumSelfAttention", "Hybrid", "HybridAttentionQCNet"]
