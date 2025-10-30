"""Hybrid quantum‑classical autoencoder using PennyLane and Qiskit."""
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.optimize import AdamOptimizer
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals


class QuantumAutoencoderGen353:
    """Variational autoencoder built on a hybrid PennyLane/Qiskit stack."""
    def __init__(
        self,
        num_features: int,
        latent_dim: int,
        *,
        device: str = "default.qubit",
        shots: int = 1024,
        learning_rate: float = 0.01,
        max_iter: int = 200,
        tol: float = 1e-3,
    ):
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.total_wires = num_features + latent_dim
        self.device = qml.device(device, wires=self.total_wires, shots=shots)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self._build_circuit()

    def _build_circuit(self):
        """Construct the variational circuit with encoding, ansatz, and decoding."""
        @qml.qnode(self.device, interface="torch")
        def circuit(inputs: np.ndarray, weights: np.ndarray):
            # Angle‑encoding of classical data
            for i in range(self.num_features):
                qml.RY(inputs[i], wires=i)

            # Variational ansatz on all qubits
            qml.apply(RealAmplitudes(wires=range(self.total_wires), reps=2))

            # Decoding: measure first num_features qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_features)]

        self.circuit = circuit

    def loss(self, batch: np.ndarray, weights: np.ndarray) -> float:
        """Mean squared error between input and reconstructed outputs."""
        recon = self.circuit(batch, weights)
        return np.mean((recon - batch) ** 2)

    def train(self, data: np.ndarray):
        """Train the variational autoencoder with Adam optimizer."""
        weights = self.circuit.default_weights
        opt = AdamOptimizer(self.learning_rate)

        for epoch in range(self.max_iter):
            loss_val = self.loss(data, weights)
            weights = opt.step(lambda w: self.loss(data, w), weights)
            if epoch % 10 == 0 or epoch == self.max_iter - 1:
                print(f"Epoch {epoch:04d} | Loss: {loss_val:.6f}")

            # Convergence check
            if loss_val < self.tol:
                print(f"Converged at epoch {epoch}")
                break

        self.weights = weights
        return weights

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode classical data into the latent subspace via measurement."""
        # Use the trained circuit to obtain latent representation
        latent = []
        for sample in data:
            probs = self.circuit(sample, self.weights)
            latent.append(probs)
        return np.array(latent)

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """Decode latent representation back to feature space."""
        # Simple linear decoder as a placeholder
        return latent @ np.eye(self.num_features)

    def full_autoencoder(self, data: np.ndarray) -> np.ndarray:
        """Run full encode–decode pipeline."""
        return self.decode(self.encode(data))


def quantum_autoencoder_factory(
    num_features: int,
    latent_dim: int,
    *,
    device: str = "default.qubit",
    shots: int = 1024,
    learning_rate: float = 0.01,
    max_iter: int = 200,
    tol: float = 1e-3,
) -> QuantumAutoencoderGen353:
    """Convenience constructor mirroring the classical factory."""
    return QuantumAutoencoderGen353(
        num_features,
        latent_dim,
        device=device,
        shots=shots,
        learning_rate=learning_rate,
        max_iter=max_iter,
        tol=tol,
    )


# Utility: domain‑wall and swap‑test construction for pedagogical experiments
def domain_wall_circuit(num_qubits: int, a: int, b: int) -> QuantumCircuit:
    """Insert X gates on qubits from a to b (exclusive)."""
    qr = QuantumRegister(num_qubits)
    qc = QuantumCircuit(qr)
    for i in range(a, b):
        qc.x(qr[i])
    return qc


def swap_test_circuit(num_qubits: int, target: int) -> QuantumCircuit:
    """Construct a swap‑test circuit on the target qubit."""
    qr = QuantumRegister(num_qubits)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr, cr)
    qc.h(qr[target])
    for i in range(num_qubits):
        if i!= target:
            qc.cswap(qr[target], qr[i], qr[(i + 1) % num_qubits])
    qc.h(qr[target])
    qc.measure(qr[target], cr[0])
    return qc


__all__ = [
    "QuantumAutoencoderGen353",
    "quantum_autoencoder_factory",
    "domain_wall_circuit",
    "swap_test_circuit",
]
