"""Quantum implementation of a hybrid auto‑sampler.

This module mirrors the classical ``HybridAutoSampler`` above but
constructs a parameterized quantum circuit using Qiskit's ``SamplerQNN``.
The circuit combines a Real‑Amplitudes ansatz for latent encoding
with a domain‑wall swap‑test to create a disentangled measurement
distribution.  The class name matches the classical counterpart so
users can interchange the two back‑ends without changing client code.
"""

from __future__ import annotations

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler

from typing import Iterable, Tuple

# ------------------------------------------------------------------
# Helper functions – domain wall & ansatz
# ------------------------------------------------------------------
def _domain_wall(qc: QuantumCircuit, start: int, end: int) -> QuantumCircuit:
    """Apply X gates to qubits in the range [start, end)."""
    for i in range(start, end):
        qc.x(i)
    return qc

def _real_amplitudes_ansatz(num_qubits: int, reps: int = 5) -> QuantumCircuit:
    return RealAmplitudes(num_qubits, reps=reps)

# ------------------------------------------------------------------
# Quantum hybrid sampler class
# ------------------------------------------------------------------
class HybridAutoSampler:
    """
    Quantum counterpart to the classical ``HybridAutoSampler``.
    Builds a SamplerQNN that samples from a latent distribution
    defined by a Real‑Amplitudes circuit and a swap‑test domain wall.
    """

    def __init__(
        self,
        latent_dim: int = 3,
        trash_dim: int = 2,
        reps: int = 5,
    ) -> None:
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.reps = reps
        self._build_circuit()

    def _build_circuit(self) -> None:
        """Construct the full quantum circuit used for sampling."""
        num_qubits = self.latent_dim + 2 * self.trash_dim + 1  # +1 auxiliary
        qr = QuantumRegister(num_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode latent + trash with ansatz
        qc.compose(_real_amplitudes_ansatz(self.latent_dim + self.trash_dim, self.reps),
                   range(0, self.latent_dim + self.trash_dim), inplace=True)

        # Swap‑test domain wall on auxiliary qubit
        aux = self.latent_dim + 2 * self.trash_dim
        qc.h(aux)
        for i in range(self.trash_dim):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.trash_dim + i)
        qc.h(aux)

        qc.measure(aux, cr[0])

        # Optional domain wall pre‑processing
        qc = _domain_wall(qc, 0, num_qubits)

        self.circuit = qc

        # Define SamplerQNN
        self.sampler = Sampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,  # identity interpret
            output_shape=2,
            sampler=self.sampler,
        )

    def sample(self, n_shots: int = 1024) -> Iterable[int]:
        """Return a list of sampled bitstrings from the quantum sampler."""
        return self.qnn.sample(n_shots)

    def get_parameters(self) -> Tuple:
        """Return current trainable parameters of the ansatz."""
        return tuple(self.qnn.parameters)

    def set_parameters(self, params: Iterable[float]) -> None:
        """Update the parameters of the ansatz."""
        self.qnn.set_parameters(params)

__all__ = ["HybridAutoSampler"]
