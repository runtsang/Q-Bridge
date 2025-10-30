"""FastBaseEstimator__gen082.py – Quantum estimator and circuit helpers.

Features
--------
* Parameter‑binding and expectation evaluation for arbitrary Qiskit BaseOperator observables.
* Shot‑noise simulation using Gaussian perturbation on real parts.
* Re‑usable auto‑encoder and classifier circuit factories.
* Mirrors the classical API for seamless switching.

Classes
-------
FastBaseEstimator
    Core evaluator for a qiskit.circuit.QuantumCircuit.
FastEstimator
    Subclass adding shot‑noise control.

Functions
---------
autoencoder_circuit
    Builds a parameterized auto‑encoder with a swap‑test readout.
build_classifier_circuit
    Constructs a layered ansatz with explicit encoding and variational parameters.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.sparse_pauli_op import SparsePauliOp

import numpy as np


class FastBaseEstimator:
    """Evaluate expectation values for a parametrised quantum circuit."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Return a 2‑D list of expectation values."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Adds optional shot‑noise to expectation values."""

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [
                complex(
                    rng.normal(mean.real, max(1e-6, 1 / shots)),
                    rng.normal(mean.imag, max(1e-6, 1 / shots)),
                )
                for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


# --------------------------------------------------------------------------- #
# Quantum auto‑encoder circuit
# --------------------------------------------------------------------------- #

def autoencoder_circuit(
    num_latent: int,
    num_trash: int,
) -> QuantumCircuit:
    """Return a circuit that encodes the input into a latent subspace.

    The circuit uses a RealAmplitudes ansatz followed by a swap‑test readout.
    """
    qr = QuantumCircuit(num_latent + 2 * num_trash + 1, name="ae")
    # Ansatz
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    qr.append(ansatz, range(0, num_latent + num_trash))
    qr.barrier()
    # Swap‑test auxiliary qubit
    aux = num_latent + 2 * num_trash
    qr.h(aux)
    for i in range(num_trash):
        qr.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qr.h(aux)
    qr.measure(aux, 0)
    return qr


# --------------------------------------------------------------------------- #
# Quantum classifier circuit
# --------------------------------------------------------------------------- #

def build_classifier_circuit(
    num_qubits: int,
    depth: int,
) -> tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], list[SparsePauliOp]]:
    """Return a layered encoding ansatz with metadata mirroring the classical API."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    index = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables


__all__ = [
    "FastBaseEstimator",
    "FastEstimator",
    "autoencoder_circuit",
    "build_classifier_circuit",
]
