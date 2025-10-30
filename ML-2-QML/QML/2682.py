"""Quantum autoencoder with attention‑style parameterization."""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

algorithm_globals.random_seed = 42
sampler = Sampler()


class QuantumAttentionBlock:
    """Quantum sub‑circuit mimicking self‑attention via rotations and controlled‑X entanglement."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        # Rotation block
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        # Entanglement block (controlled‑X)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ):
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)


class Gen250AutoencoderQNN:
    """Variational quantum autoencoder that embeds an attention‑like block."""

    def __init__(self, latent_dim: int = 3, trash_dim: int = 2):
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.total_qubits = latent_dim + 2 * trash_dim + 1
        self.attention = QuantumAttentionBlock(n_qubits=latent_dim + trash_dim)

    def _autoencoder_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        qr = QuantumRegister(self.total_qubits, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)

        # Encode part – RealAmplitudes ansatz
        ansatz = RealAmplitudes(self.latent_dim + self.trash_dim, reps=5)
        circuit.compose(ansatz, range(0, self.latent_dim + self.trash_dim), inplace=True)

        # Swap‑test style measurement
        auxiliary = self.latent_dim + 2 * self.trash_dim
        circuit.h(auxiliary)
        for i in range(self.trash_dim):
            circuit.cswap(auxiliary, self.latent_dim + i, self.latent_dim + self.trash_dim + i)
        circuit.h(auxiliary)
        circuit.measure(auxiliary, cr[0])

        # Attention block on the remaining qubits
        attention_circ = self.attention._build_circuit(rotation_params, entangle_params)
        circuit.compose(attention_circ, range(0, self.latent_dim + self.trash_dim), inplace=True)

        return circuit

    def build_qnn(self) -> SamplerQNN:
        """Return a SamplerQNN that can be trained with classical optimizers."""
        # Dummy parameter arrays for the interface; actual values are optimized during training
        rotation_params = np.zeros(3 * (self.latent_dim + self.trash_dim))
        entangle_params = np.zeros(self.latent_dim + self.trash_dim - 1)

        qc = self._autoencoder_circuit(rotation_params, entangle_params)
        qnn = SamplerQNN(
            circuit=qc,
            input_params=[],
            weight_params=qc.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=sampler,
        )
        return qnn


def Gen250AutoencoderQNN(latent_dim: int = 3, trash_dim: int = 2) -> SamplerQNN:
    """Convenience factory mirroring the classical counterpart."""
    return Gen250AutoencoderQNN(latent_dim, trash_dim).build_qnn()


__all__ = ["Gen250AutoencoderQNN", "QuantumAttentionBlock", "Gen250AutoencoderQNN"]
