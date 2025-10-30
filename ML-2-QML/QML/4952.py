"""Hybrid quantum autoencoder with self-attention and kernel subcircuits."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler, StatevectorSampler
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector


# ---------- Helper: Self‑Attention ----------
class QuantumSelfAttention:
    """Quantum implementation of a self‑attention block."""
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

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
        circuit = self._build_circuit(rotation_params, entangle_params)
        backend = Aer.get_backend("qasm_simulator")
        job = execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)


# ---------- Helper: Kernel ----------
def quantum_kernel(x: np.ndarray, y: np.ndarray, n_qubits: int = 4) -> float:
    """Evaluate a simple quantum kernel using Ry encodings and state‑vector overlap."""
    qr = QuantumRegister(n_qubits, "q")
    qc = QuantumCircuit(qr)
    for i in range(n_qubits):
        qc.ry(x[i], i)
    for i in range(n_qubits):
        qc.ry(-y[i], i)
    sim = AerSimulator(method="statevector")
    result = sim.run([qc]).result()
    state = result.get_statevector()
    return np.abs(state[0]) ** 2


# ---------- Core hybrid quantum autoencoder ----------
class HybridAutoencoder:
    """Hybrid quantum autoencoder that fuses variational encoding, quantum self‑attention, and a quantum kernel."""
    def __init__(self, input_dim: int, latent_dim: int = 3, trash: int = 2):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.trash = trash
        self.sampler = Sampler()
        self._build_circuits()

    def _build_circuits(self):
        # Variational encoder
        self.encoder = RealAmplitudes(self.latent_dim + self.trash, reps=5)
        # Self‑attention
        self.attn = QuantumSelfAttention(n_qubits=self.latent_dim)
        # Decoder (classical linear layer)
        self.decoder_weights = np.random.randn(self.latent_dim, self.input_dim)

    def _encode_circuit(self, x: np.ndarray) -> QuantumCircuit:
        """Build a circuit that encodes a single sample into the latent register."""
        qr = QuantumRegister(self.latent_dim + 2 * self.trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        # Feature map: simple Ry encoding on first input_dim qubits
        for i in range(min(self.input_dim, qr.size)):
            circuit.ry(x[i], i)
        # Variational part
        circuit.append(self.encoder, range(self.latent_dim + self.trash))
        # Swap‑test for latent extraction
        aux = self.latent_dim + 2 * self.trash
        circuit.h(aux)
        for i in range(self.trash):
            circuit.cswap(aux, self.latent_dim + i, self.latent_dim + self.trash + i)
        circuit.h(aux)
        circuit.measure(aux, cr)
        return circuit

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode a batch of data to latent vectors using the quantum autoencoder."""
        latent_vectors = []
        for sample in data:
            circ = self._encode_circuit(sample)
            result = self.sampler.run(circ).result()
            counts = result.get_counts(circ)
            # Convert measurement outcomes to a latent representation
            bit = max(counts, key=counts.get)
            latent = np.array([int(bit)])
            latent_vectors.append(latent)
        return np.vstack(latent_vectors)

    def decode(self, latents: np.ndarray) -> np.ndarray:
        """Classical linear decoder mapping latent vectors back to input space."""
        return latents @ self.decoder_weights

    def forward(self, data: np.ndarray) -> np.ndarray:
        """Full encode‑decode pipeline."""
        z = self.encode(data)
        return self.decode(z)

    def kernel_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute a Gram matrix using the quantum kernel."""
        return np.array([[quantum_kernel(x, y) for y in b] for x in a])


__all__ = ["HybridAutoencoder"]
