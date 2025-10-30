import numpy as np
import qiskit as qk
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes, ZGate
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.kernels import QuantumKernel as QiskitQuantumKernel
from qiskit.providers.aer import AerSimulator
from typing import Sequence

class QuantumKernel:
    """Quantum kernel via a variational ansatz and swap test."""
    def __init__(self, num_qubits: int = 4):
        self.num_qubits = num_qubits
        self.backend = AerSimulator(method='statevector')
        self.feature_map = RealAmplitudes(num_qubits, reps=2)
        self.qk = QiskitQuantumKernel(
            feature_map=self.feature_map,
            quantum_instance=self.backend
        )

    def forward(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return kernel value for two classical vectors."""
        return float(self.qk.evaluate(x, y))

class QuantumAutoencoder:
    """
    Minimal quantum autoencoder that encodes a classical vector into a latent
    representation consisting of expectation values of Z on a subset of qubits.
    """
    def __init__(self, num_latent: int = 3, num_trash: int = 2):
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.backend = AerSimulator(method='statevector')

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Return a latent vector of length ``num_latent``."""
        if len(x) < self.num_latent:
            raise ValueError("Input dimension must be at least ``num_latent``.")
        qc = QuantumCircuit(self.num_latent)
        for i, val in enumerate(x[:self.num_latent]):
            qc.ry(val, i)
        sv = Statevector.from_instruction(qc)
        return np.array([sv.expectation_value(ZGate(), [i]) for i in range(self.num_latent)])

class UnifiedKernelAutoencoder:
    """
    Hybrid quantum‑classical kernel autoencoder.
    Encodes data via a quantum autoencoder, builds a classical RBF kernel on the latent space,
    and also provides a pure quantum kernel for direct similarity.
    """
    def __init__(
        self,
        num_latent: int = 3,
        num_trash: int = 2,
        num_qubits: int = 4,
        latent_gamma: float = 1.0,
    ):
        self.autoencoder = QuantumAutoencoder(num_latent, num_trash)
        self.quantum_kernel = QuantumKernel(num_qubits)
        self.latent_gamma = latent_gamma

    @staticmethod
    def _rbf(a: np.ndarray, b: np.ndarray, gamma: float) -> float:
        diff = a - b
        return np.exp(-gamma * np.dot(diff, diff))

    def encode(self, x: np.ndarray) -> np.ndarray:
        return self.autoencoder.encode(x)

    def decode(self, z: np.ndarray) -> np.ndarray:
        # Placeholder: in a real implementation this would reconstruct the input.
        return z

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(x))

    def compute_quantum_kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        return self.quantum_kernel.forward(x, y)

    def compute_latent_kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        z_x = self.encode(x)
        z_y = self.encode(y)
        return self._rbf(z_x, z_y, self.latent_gamma)

    def compute_combined_kernel(
        self, x: np.ndarray, y: np.ndarray, alpha: float = 0.5
    ) -> float:
        return alpha * self.compute_quantum_kernel(x, y) + (1 - alpha) * self.compute_latent_kernel(x, y)

    def kernel_matrix(
        self,
        a: Sequence[np.ndarray],
        b: Sequence[np.ndarray],
        alpha: float = 0.5,
    ) -> np.ndarray:
        mat = np.empty((len(a), len(b)), dtype=np.float64)
        for i, xi in enumerate(a):
            for j, yj in enumerate(b):
                mat[i, j] = self.compute_combined_kernel(xi, yj, alpha)
        return mat

__all__ = [
    "QuantumKernel",
    "QuantumAutoencoder",
    "UnifiedKernelAutoencoder",
]
