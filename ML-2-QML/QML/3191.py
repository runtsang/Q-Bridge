"""Hybrid quantum autoencoder that embeds a quantum self‑attention subcircuit into the variational encoder.

The module defines:
- QuantumSelfAttention: builds a self‑attention style circuit with Rx/Ry/Rz rotations and controlled‑Rx entanglement.
- HybridQuantumAutoencoder: a variational encoder using RealAmplitudes followed by QuantumSelfAttention, then produces a classical latent vector via statevector sampling.
- factory function `HybridAutoencoderFactory` mirroring the classical API.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN

class QuantumSelfAttention:
    """Quantum self‑attention subcircuit with parameterized rotations and controlled‑Rx entanglement."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, 'q')
        self.cr = ClassicalRegister(n_qubits, 'c')

    def build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        # local rotations
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        # pairwise entanglement
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, backend, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
        circuit = self.build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)

class HybridQuantumAutoencoder:
    """Variational quantum autoencoder with a quantum self‑attention subcircuit."""
    def __init__(self, latent_dim: int, trash_dim: int = 2):
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.num_qubits = latent_dim + 2 * trash_dim + 1  # auxiliary qubit for swap‑test
        self.qr = QuantumRegister(self.num_qubits, 'q')
        self.circuit = QuantumCircuit(self.qr)

        # ansatz
        ansatz = RealAmplitudes(self.latent_dim + self.trash_dim, reps=5)
        self.circuit.compose(ansatz, range(0, self.latent_dim + self.trash_dim), inplace=True)

        # swap test with auxiliary qubit
        aux = self.latent_dim + 2 * self.trash_dim
        self.circuit.h(aux)
        for i in range(self.trash_dim):
            self.circuit.cswap(aux, self.latent_dim + i, self.latent_dim + self.trash_dim + i)
        self.circuit.h(aux)

        # quantum self‑attention subcircuit
        self.attention = QuantumSelfAttention(self.latent_dim)
        # initialize with random parameters; in practice these would be trainable
        rot_params = np.random.rand(3 * self.latent_dim)
        ent_params = np.random.rand(self.latent_dim - 1)
        att_circ = self.attention.build_circuit(rot_params, ent_params)
        self.circuit.compose(att_circ, inplace=True)

        # Sampler for statevector
        self.sampler = Aer.get_backend('statevector_simulator')
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=ansatz.parameters,
            interpret=lambda sv: np.real(sv.data)[:self.latent_dim],
            output_shape=(self.latent_dim,),
            sampler=self.sampler,
        )

    def encode(self, input_vector: np.ndarray) -> np.ndarray:
        """Run the variational circuit and return a classical latent vector."""
        # The input_vector is not encoded in this toy example; real applications would use a feature map.
        latent = self.qnn.forward(np.array([]))
        return latent

    def decode(self, latent_vector: np.ndarray) -> np.ndarray:
        """Placeholder: map classical latent back to reconstructed data."""
        return latent_vector  # identity for illustration

    def forward(self, input_vector: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(input_vector))

def HybridAutoencoderFactory(
    latent_dim: int,
    *,
    trash_dim: int = 2,
) -> HybridQuantumAutoencoder:
    """Return a configured :class:`HybridQuantumAutoencoder`."""
    return HybridQuantumAutoencoder(latent_dim=latent_dim, trash_dim=trash_dim)

__all__ = ["HybridAutoencoder", "HybridQuantumAutoencoder", "QuantumSelfAttention"]
