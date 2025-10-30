"""Hybrid quantum classifier with data‑re‑uploading and shot‑noise simulation."""

from __future__ import annotations

from typing import Iterable, Sequence, List

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class QuantumClassifierModel:
    """Quantum circuit that emulates a classical classifier interface.

    The ansatz supports data‑re‑uploading, configurable entanglement, and
    optional shot‑noise emulation.  It can be evaluated either with a
    state‑vector simulator or a shot‑based simulator.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int = 3,
        encoding: str = "rx",
        entanglement: str = "cz",
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.encoding = encoding
        self.entanglement = entanglement
        self.shots = shots
        self.seed = seed

        self._circuit, self._encoding_params, self._weight_params, self._observables = build_classifier_circuit(
            num_qubits, depth, encoding, entanglement
        )
        self._simulator = Aer.get_backend("aer_simulator_statevector") if shots is None else Aer.get_backend("aer_simulator")

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._encoding_params) + len(self._weight_params):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._encoding_params + self._weight_params, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound = self._bind(values)
            if self.shots is None:
                state = Statevector.from_instruction(bound)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                job = execute(bound, self._simulator, shots=self.shots, seed_simulator=self.seed)
                result = job.result()
                counts = result.get_counts()
                row = []
                for obs in observables:
                    exp = 0.0
                    for bitstring, count in counts.items():
                        z = 1 if bitstring[-1] == '1' else -1
                        exp += z * count
                    exp /= self.shots
                    row.append(exp)
            results.append(row)
        return results


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    encoding: str = "rx",
    entanglement: str = "cz",
) -> tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
    """Construct a layered ansatz with explicit encoding and variational parameters.

    Parameters
    ----------
    num_qubits:
        Number of qubits / input features.
    depth:
        Number of variational layers.
    encoding:
        Gate used for data encoding ("rx" or "ry").
    entanglement:
        Type of entangling gate ("cz" or "cx").
    """
    enc_params = ParameterVector("x", num_qubits)
    weight_params = ParameterVector("theta", num_qubits * depth)
    circuit = QuantumCircuit(num_qubits)
    # Initial encoding
    for idx, qubit in enumerate(range(num_qubits)):
        if encoding == "rx":
            circuit.rx(enc_params[idx], qubit)
        else:
            circuit.ry(enc_params[idx], qubit)

    # Variational layers with data re‑uploading
    for l in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weight_params[l * num_qubits + qubit], qubit)
        if entanglement == "cz":
            for qubit in range(num_qubits - 1):
                circuit.cz(qubit, qubit + 1)
        else:
            for qubit in range(num_qubits - 1):
                circuit.cx(qubit, qubit + 1)
        # Re‑upload data
        for idx, qubit in enumerate(range(num_qubits)):
            if encoding == "rx":
                circuit.rx(enc_params[idx], qubit)
            else:
                circuit.ry(enc_params[idx], qubit)

    # Observables: Z on each qubit
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(enc_params), list(weight_params), observables


__all__ = ["QuantumClassifierModel", "build_classifier_circuit"]
