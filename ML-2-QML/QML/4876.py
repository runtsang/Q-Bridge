"""
Hybrid Autoencoder – quantum implementation using Qiskit.
"""

from __future__ import annotations

import math
from typing import Iterable, List

import numpy as np
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

# --------------------------------------------------------------------------- #
# Helper utilities – FastBaseEstimator style
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametric circuit."""

    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Iterable[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[Statevector],
        parameter_sets: Iterable[Iterable[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


# --------------------------------------------------------------------------- #
# Variational autoencoder circuit
# --------------------------------------------------------------------------- #
def _real_amplitudes_ansatz(num_qubits: int, reps: int = 5) -> QuantumCircuit:
    """Return a RealAmplitudes ansatz circuit."""
    return RealAmplitudes(num_qubits, reps=reps)


def _swap_test_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """
    Build a swap‑test based encoder that pairs latent qubits with trash qubits.
    Returns a circuit that measures the overlap between the latent state and a
    reference |0> state on an auxiliary qubit.
    """
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)

    # Encode latent part
    circuit.append(_real_amplitudes_ansatz(num_latent + num_trash), range(0, num_latent + num_trash))
    circuit.barrier()

    aux = num_latent + 2 * num_trash
    circuit.h(aux)
    for i in range(num_trash):
        circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
    circuit.h(aux)
    circuit.measure(aux, cr[0])
    return circuit


class HybridAutoencoderQML:
    """
    Quantum autoencoder that maps a classical latent vector to a quantum state
    and decodes it back via a sampler‑based variational circuit.
    """

    def __init__(
        self,
        latent_dim: int,
        trash_dim: int = 2,
        num_qubits: int | None = None,
        shots: int = 100,
    ) -> None:
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.num_qubits = num_qubits or (latent_dim + 2 * trash_dim + 1)
        self.shots = shots

        # Build the full auto‑encoder circuit
        self.circuit = _swap_test_circuit(latent_dim, trash_dim)

        # Sampler for expectation values
        self.sampler = StatevectorSampler()
        # Interpret as raw amplitude (no scaling)
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )

    def encode(self, latent: np.ndarray) -> np.ndarray:
        """
        Run the quantum encoder on a single latent vector.
        Returns the probability of measuring |1> on the auxiliary qubit.
        """
        if latent.size!= self.latent_dim:
            raise ValueError("Incorrect latent dimension.")
        # Bind latent parameters – here we simply use the latent values as rotation angles
        bound_circuit = self.circuit.assign_parameters(
            dict(zip(self.circuit.parameters, latent)), inplace=False
        )
        result = self.sampler.run(bound_circuit, shots=self.shots).get_counts(self.circuit)
        # Compute average probability of |1>
        counts = sum(int(bit) for key, val in result.items() for bit in key)
        return np.array([counts / (self.shots * self.num_qubits)])

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """
        Variational decoder that reconstructs a classical vector from the
        quantum state.  Uses the sampler to obtain expectation values for
        a set of Pauli‑Z observables on the latent qubits.
        """
        paulis = [Statevector.from_label("0" * self.latent_dim)]
        results = self.qnn.evaluate(latent, paulis)
        return np.array(results[0])

    def forward(self, latent: np.ndarray) -> np.ndarray:
        """Encode‑then‑decode pipeline."""
        return self.decode(self.encode(latent))

    def evaluate(
        self,
        observables: Iterable[Statevector],
        parameter_sets: Iterable[Iterable[float]],
    ) -> List[List[complex]]:
        """Convenience wrapper around FastBaseEstimator for external use."""
        estimator = FastBaseEstimator(self.circuit)
        return estimator.evaluate(observables, parameter_sets)

__all__ = [
    "HybridAutoencoderQML",
    "FastBaseEstimator",
]
