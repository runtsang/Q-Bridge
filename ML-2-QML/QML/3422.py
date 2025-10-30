"""Quantum autoencoder implementation using Qiskit and SamplerQNN.

The module builds a variational circuit that encodes classical data into a
low‑dimensional latent space via a swap‑test style measurement.  It also
provides a FastBaseEstimator for evaluating expectation values of arbitrary
observables, mirroring the classical FastEstimator pattern.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals

# ---------------------------------------------------------------------
# FastBaseEstimator (quantum version)
# ---------------------------------------------------------------------


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrised circuit."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[object], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

# ---------------------------------------------------------------------
# Quantum autoencoder circuit
# ---------------------------------------------------------------------


def domain_wall(circuit: QuantumCircuit, a: int, b: int) -> QuantumCircuit:
    """Insert an X‑domain wall between qubits a and b."""
    for i in range(a, b):
        circuit.x(i)
    return circuit


def quantum_autoencoder_circuit(num_latent: int, num_trash: int, reps: int = 5) -> QuantumCircuit:
    """Build a variational circuit that maps classical data to a latent space.

    Parameters
    ----------
    num_latent : int
        Size of the latent register.
    num_trash : int
        Number of trash qubits used for swap‑test.
    reps : int
        Number of repetitions for RealAmplitudes ansatz.
    """
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, name="q")
    cr = ClassicalRegister(1, name="c")
    circuit = QuantumCircuit(qr, cr)

    # Variational ansatz on the first (num_latent + num_trash) qubits
    ansatz = RealAmplitudes(num_latent + num_trash, reps=reps)
    circuit.compose(ansatz, range(0, num_latent + num_trash), inplace=True)
    circuit.barrier()

    # Swap‑test to entangle trash qubits with auxiliary qubit
    aux = num_latent + 2 * num_trash
    circuit.h(aux)
    for i in range(num_trash):
        circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
    circuit.h(aux)
    circuit.measure(aux, cr[0])

    return circuit


# ---------------------------------------------------------------------
# SamplerQNN wrapper
# ---------------------------------------------------------------------


def identity_interpret(x: np.ndarray) -> np.ndarray:
    """Return measurement probabilities unchanged."""
    return x


def autoencoder_qnn(num_latent: int = 3, num_trash: int = 2, reps: int = 5) -> SamplerQNN:
    """Instantiate a SamplerQNN that implements the quantum autoencoder."""
    algorithm_globals.random_seed = 42
    sampler = algorithm_globals.sampler

    circ = quantum_autoencoder_circuit(num_latent, num_trash, reps=reps)
    qnn = SamplerQNN(
        circuit=circ,
        input_params=[],
        weight_params=circ.parameters,
        interpret=identity_interpret,
        output_shape=2,
        sampler=sampler,
    )
    return qnn


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

__all__ = [
    "FastBaseEstimator",
    "quantum_autoencoder_circuit",
    "domain_wall",
    "autoencoder_qnn",
    "identity_interpret",
]
