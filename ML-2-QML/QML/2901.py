"""Hybrid kernel module using Qiskit.

The module implements:
* `HybridKernel` – a weighted sum of a classical RBF kernel and a
  variational quantum kernel built with Qiskit.  The quantum part
  evaluates the squared overlap between two statevectors prepared
  with Ry rotations.
* `HybridFCL` – a fully‑connected layer that can be classical
  (torch.nn.Linear) or quantum (a single‑qubit Ry circuit executed
  on an Aer simulator).  The `run` method accepts an iterable of
  angles and returns a NumPy array.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector
from typing import Sequence, Iterable

# Classical RBF kernel
class RBFAnsatz:
    def __init__(self, gamma: float = 1.0) -> None:
        self.gamma = gamma

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        diff = x - y
        return float(np.exp(-self.gamma * np.sum(diff * diff)))

# Quantum kernel
class QuantumAnsatz:
    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits

    def _statevector(self, params: np.ndarray) -> Statevector:
        qc = QuantumCircuit(self.n_qubits)
        for i, theta in enumerate(params):
            qc.ry(theta, i)
        return Statevector.from_instruction(qc)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        sv_x = self._statevector(x)
        sv_y = self._statevector(y)
        overlap = np.abs(sv_x.inner(sv_y)) ** 2
        return float(overlap)

class HybridKernel:
    def __init__(self, gamma: float = 1.0, alpha: float = 0.5, n_qubits: int = 4) -> None:
        self.alpha = alpha
        self.classical = RBFAnsatz(gamma)
        self.quantum = QuantumAnsatz(n_qubits)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        cls = self.classical(x, y)
        qk = self.quantum(x, y)
        return self.alpha * cls + (1.0 - self.alpha) * qk

def kernel_matrix(a: Sequence[np.ndarray], b: Sequence[np.ndarray],
                  gamma: float = 1.0, alpha: float = 0.5, n_qubits: int = 4) -> np.ndarray:
    kernel = HybridKernel(gamma, alpha, n_qubits)
    return np.array([[kernel(x, y) for y in b] for x in a])

# Hybrid fully‑connected layer
class HybridFCL:
    def __init__(self, n_features: int = 1, use_quantum: bool = False) -> None:
        self.use_quantum = use_quantum
        if self.use_quantum:
            self.n_qubits = 1
            self.backend = Aer.get_backend("qasm_simulator")
            self.shots = 1024
        else:
            self.n_features = n_features

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        if self.use_quantum:
            qc = QuantumCircuit(1)
            qc.h(0)
            for theta in thetas:
                qc.ry(theta, 0)
            qc.measure_all()
            job = execute(qc, self.backend, shots=self.shots)
            result = job.result()
            counts = result.get_counts(qc)
            expectation = 0.0
            for state, count in counts.items():
                expectation += int(state, 2) * count
            expectation /= self.shots
            return np.array([expectation])
        else:
            values = np.array(list(thetas), dtype=np.float32).reshape(-1, 1)
            # Simple linear transform followed by tanh
            weights = np.random.randn(self.n_features, 1).astype(np.float32)
            bias = np.random.randn(1).astype(np.float32)
            linear = values @ weights + bias
            expectation = np.tanh(linear).mean()
            return np.array([expectation])

__all__ = ["HybridKernel", "kernel_matrix", "HybridFCL"]
