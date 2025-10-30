"""Hybrid autoencoder – quantum encoder implementation.

The quantum encoder maps a classical latent vector into a parameterised
real‑amplitudes circuit.  A swap‑test style decoder is emulated by
sampling the statevector and computing the probability of each qubit
being in |1⟩.  This module exposes the same class name and method
signatures as the classical counterpart so the two can be swapped
in a hybrid pipeline.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector

# The following imports are optional but provide a convenient
# interface for gradient‑based optimisation if desired.
# from qiskit_machine_learning.neural_networks import SamplerQNN
# from qiskit_machine_learning.optimizers import COBYLA
# from qiskit_machine_learning.utils import algorithm_globals


class HybridAutoencoderQuantum:
    """Quantum encoder for a hybrid autoencoder."""
    def __init__(self, latent_dim: int, num_qubits: int | None = None):
        self.latent_dim = latent_dim
        self.num_qubits = num_qubits or latent_dim
        self.sampler = Sampler()
        self.circuit = self._build_circuit()
        # The sampler automatically takes care of parameter binding.
        # For a full hybrid training loop one would wrap this in a
        # SamplerQNN and attach a classical optimiser.

    def _build_circuit(self) -> QuantumCircuit:
        """Build a RealAmplitudes ansatz that accepts `latent_dim` parameters."""
        qr = QuantumRegister(self.num_qubits, "q")
        qc = QuantumCircuit(qr)
        # The ansatz will be parameterised by the latent vector.
        ansatz = RealAmplitudes(self.num_qubits, reps=2)
        qc.compose(ansatz, inplace=True)
        return qc

    def encode(self, latent: np.ndarray) -> dict:
        """
        Encode a latent vector into a set of circuit parameters.
        The vector is normalised to the interval [-π, π] and then
        mapped one‑to‑one to the circuit parameters.
        """
        theta = np.clip(latent / (np.linalg.norm(latent) + 1e-8), -1.0, 1.0) * np.pi
        return dict(zip(self.circuit.parameters, theta))

    def decode(self, params: dict) -> np.ndarray:
        """
        Decode the quantum state by sampling the statevector and
        computing the probability of each qubit being in |1⟩.
        """
        result = self.sampler.run(self.circuit, parameter_binds=[params]).result()
        state = Statevector(result.get_statevector())
        probs = state.probabilities()
        p1 = []
        for qubit in range(self.num_qubits):
            # Sum probabilities where the qubit is |1⟩
            mask = 1 << qubit
            p1_i = sum(p for idx, p in enumerate(probs) if idx & mask)
            p1.append(p1_i)
        return np.array(p1)

    def forward(self, latent: np.ndarray) -> np.ndarray:
        """Full forward pass: encode → sample → decode."""
        params = self.encode(latent)
        return self.decode(params)


def HybridAutoencoderQuantumFactory(
    latent_dim: int,
    num_qubits: int | None = None,
) -> HybridAutoencoderQuantum:
    """Return a ready‑to‑use quantum encoder."""
    return HybridAutoencoderQuantum(latent_dim, num_qubits)
