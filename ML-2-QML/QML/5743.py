"""Quantum estimator for a parametrized circuit with gradient support.

`FastBaseEstimatorGen128` evaluates expectation values of given
observables, optionally simulating shot noise and providing a
parameter‑shift gradient routine that can be used for variational
optimization.

The implementation uses Qiskit’s Statevector for exact evaluation
and the Aer simulator for noisy shot simulation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import Aer
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

class FastBaseEstimatorGen128:
    """Evaluate expectation values of observables for a parametrized circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parametrized variational circuit.
    device : str, optional
        Backend:'statevector' for exact evaluation, 'qasm' for shot noise.
    shots : int, optional
        Number of shots for the qasm backend.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        device: str = "statevector",
        shots: int | None = None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self._device = device
        self._shots = shots
        if device == "statevector":
            self._backend = Aer.get_backend("statevector_simulator")
        elif device == "qasm":
            self._backend = Aer.get_backend("qasm_simulator")
        else:
            raise ValueError("Unsupported backend")

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
        """Compute expectation values for each parameter set and observable."""
        obs_list = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            bound_circ = self._bind(values)
            if self._device == "statevector":
                state = Statevector.from_instruction(bound_circ)
                row = [state.expectation_value(obs) for obs in obs_list]
            else:
                job = self._backend.run(bound_circ, shots=self._shots or 1024)
                result = job.result()
                counts = result.get_counts(bound_circ)
                # Convert counts to expectation values
                row = []
                for obs in obs_list:
                    exp = 0.0
                    for bitstring, count in counts.items():
                        eig = self._bitstring_to_eigenvalue(bitstring, obs)
                        exp += eig * count
                    exp /= sum(counts.values())
                    row.append(complex(exp))
            results.append(row)

        return results

    def _bitstring_to_eigenvalue(self, bitstring: str, op: BaseOperator) -> float:
        """Map a bitstring to the eigenvalue of a Pauli operator."""
        # Very simplified: only works for PauliZ operators.
        eigenvalue = 1.0
        for qubit, char in enumerate(reversed(bitstring)):
            if hasattr(op, "paulis") and op.paulis[0][qubit] == "Z":
                eigenvalue *= 1 if bitstring[-qubit - 1] == "0" else -1
        return eigenvalue

    def gradient(
        self,
        observable: BaseOperator,
        parameter_sets: Sequence[Sequence[float]],
        shift: float = np.pi,
    ) -> List[List[float]]:
        """Compute parameter‑shift gradient for a single observable."""
        gradients: List[List[float]] = []
        for params in parameter_sets:
            grad_row: List[float] = []
            for i in range(len(params)):
                shifted_plus = list(params)
                shifted_plus[i] += shift / 2
                shifted_minus = list(params)
                shifted_minus[i] -= shift / 2
                val_plus = self.evaluate([observable], [shifted_plus])[0][0].real
                val_minus = self.evaluate([observable], [shifted_minus])[0][0].real
                grad = (val_plus - val_minus) / 2
                grad_row.append(grad)
            gradients.append(grad_row)
        return gradients


__all__ = ["FastBaseEstimatorGen128"]
