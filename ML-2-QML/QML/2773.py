"""Quantum self‑attention auto‑encoder built with Qiskit.

The :class:`SelfAttentionAutoencoder` encapsulates a parameter‑driven circuit that
first applies a self‑attention style block (rotations + entanglement) and then
runs a quantum auto‑encoder (RealAmplitudes ansatz + swap test).  It exposes a
``run`` method that accepts a backend and parameter arrays, mirroring the
original SelfAttention and Autoencoder interfaces.

The implementation is intentionally lightweight and suitable for
simulation or execution on real devices via the Qiskit Aer or Braket
backends.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.primitives import StatevectorSampler as Sampler


class SelfAttentionAutoencoder:
    """Quantum self‑attention auto‑encoder circuit."""

    def __init__(self, n_qubits: int, latent_dim: int = 3, num_trash: int = 2) -> None:
        self.n_qubits = n_qubits
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        # Prepare a default backend for quick testing
        self.backend = Aer.get_backend("qasm_simulator")

    def _attention_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        """Build the self‑attention sub‑circuit."""
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        circuit = QuantumCircuit(qr, cr)

        # Rotation layer
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)

        # Entanglement layer
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)

        return circuit

    def _autoencoder_circuit(self) -> QuantumCircuit:
        """Build the quantum auto‑encoder sub‑circuit."""
        num_qubits = self.latent_dim + 2 * self.num_trash + 1
        qr = QuantumRegister(num_qubits, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)

        # Ansatz on latent + trash qubits
        circuit.compose(
            RealAmplitudes(self.latent_dim + self.num_trash, reps=5),
            range(0, self.latent_dim + self.num_trash),
            inplace=True,
        )
        circuit.barrier()

        # Swap‑test auxiliary qubit
        aux = self.latent_dim + 2 * self.num_trash
        circuit.h(aux)
        for i in range(self.num_trash):
            circuit.cswap(aux, self.latent_dim + i, self.latent_dim + self.num_trash + i)
        circuit.h(aux)
        circuit.measure(aux, cr[0])

        return circuit

    def _combined_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        """Concatenate attention and auto‑encoder circuits."""
        attn = self._attention_circuit(rotation_params, entangle_params)
        ae = self._autoencoder_circuit()
        combined = QuantumCircuit()
        combined.compose(attn, inplace=True)
        combined.compose(ae, inplace=True)
        return combined

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        """
        Execute the combined circuit on the selected backend.

        Parameters
        ----------
        rotation_params:
            Array of shape (3 * n_qubits,) containing rotation angles.
        entangle_params:
            Array of shape (n_qubits - 1,) containing CRX angles.
        shots:
            Number of measurement shots.

        Returns
        -------
        dict
            Measurement counts from the auto‑encoder auxiliary qubit.
        """
        circuit = self._combined_circuit(rotation_params, entangle_params)
        job = execute(circuit, self.backend, shots=shots)
        return job.result().get_counts(circuit)

    def sampler(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        """Return a statevector sampler for the combined circuit."""
        circuit = self._combined_circuit(rotation_params, entangle_params)
        sampler = Sampler()
        return sampler.run(circuit).result().get_statevector(circuit)


def SelfAttentionAutoencoderFactory(
    n_qubits: int,
    latent_dim: int = 3,
    num_trash: int = 2,
) -> SelfAttentionAutoencoder:
    """Convenient factory mirroring the original seed interface."""
    return SelfAttentionAutoencoder(n_qubits, latent_dim, num_trash)


__all__ = ["SelfAttentionAutoencoder", "SelfAttentionAutoencoderFactory"]
