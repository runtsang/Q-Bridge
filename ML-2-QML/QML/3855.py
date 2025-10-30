import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute

class QuantumSelfAttention:
    """Quantum self‑attention block built with Qiskit."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, backend, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)

class HybridSelfAttention:
    """Hybrid classical‑quantum self‑attention model with a CNN backbone."""
    def __init__(self, n_qubits: int = 4):
        self.attention = QuantumSelfAttention(n_qubits)
        self.backend = Aer.get_backend("qasm_simulator")
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def run(self, inputs: torch.Tensor, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024) -> torch.Tensor:
        # Classical feature extraction
        features = self.cnn(inputs)
        flattened = features.view(features.shape[0], -1).cpu().numpy()
        # Quantum attention on the flattened features
        quantum_counts = self.attention.run(self.backend, rotation_params, entangle_params, shots=shots)
        # Convert measurement counts to probabilities
        total = sum(quantum_counts.values())
        probs = np.zeros(2 ** self.attention.n_qubits)
        for state, count in quantum_counts.items():
            idx = int(state[::-1], 2)  # reverse bit order to match Qiskit output
            probs[idx] = count
        probs = probs / total
        return torch.tensor(probs, dtype=torch.float32)

__all__ = ["HybridSelfAttention"]
