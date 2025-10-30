import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from typing import Sequence, Optional

class ClassicalRBFKernel:
    """Classical RBF kernel implemented with NumPy."""
    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        diff = x - y
        return np.exp(-self.gamma * np.sum(diff ** 2))

class QuantumKernel:
    """Quantum kernel based on a parameterised Ry circuit and entangling CZ gates."""
    def __init__(self, num_qubits: int = 4):
        self.num_qubits = num_qubits
        self.base_circuit = self._build_base_circuit()

    def _build_base_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        for i in range(self.num_qubits):
            qc.ry(0.0, i)
        for i in range(self.num_qubits - 1):
            qc.cz(i, i + 1)
        return qc

    def encode(self, params: np.ndarray) -> QuantumCircuit:
        qc = self.base_circuit.copy()
        for i, theta in enumerate(params):
            qc.ry(theta, i)
        return qc

    def kernel_value(self, x: np.ndarray, y: np.ndarray) -> float:
        qc_x = self.encode(x)
        qc_y = self.encode(y)
        sv_x = Statevector.from_instruction(qc_x)
        sv_y = Statevector.from_instruction(qc_y)
        return abs(sv_x.data.conj().dot(sv_y.data))

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        return self.kernel_value(x, y)

class HybridKernel:
    """
    Unified kernel that can compute either a classical RBF kernel or a
    quantum kernel via Qiskit.  It also offers a fast evaluation routine
    with optional Gaussian shot noise, mirroring FastBaseEstimator.
    """
    def __init__(self, gamma: float = 1.0, use_quantum: bool = True, num_qubits: int = 4):
        self.gamma = gamma
        self.use_quantum = use_quantum
        if use_quantum:
            self.kernel = QuantumKernel(num_qubits=num_qubits)
        else:
            self.kernel = ClassicalRBFKernel(gamma)

    def kernel_matrix(self, a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> np.ndarray:
        """Return the Gram matrix between two batches of NumPy arrays."""
        mat = np.empty((len(a), len(b)))
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                mat[i, j] = self.kernel(x, y)
        return mat

    def evaluate(self, X: Sequence[Sequence[float]], Y: Sequence[Sequence[float]],
                 *, shots: Optional[int] = None, seed: Optional[int] = None) -> np.ndarray:
        """
        Compute the kernel matrix for two collections of data points.
        If ``shots`` is provided, Gaussian noise with variance 1/shots is added.
        """
        X_np = [np.asarray(x, dtype=np.float64) for x in X]
        Y_np = [np.asarray(y, dtype=np.float64) for y in Y]
        K = self.kernel_matrix(X_np, Y_np)
        if shots is None:
            return K
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, 1.0 / np.sqrt(shots), size=K.shape)
        return K + noise

__all__ = ["HybridKernel"]
