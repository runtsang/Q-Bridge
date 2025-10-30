"""Quantum estimator that evaluates expectation values of observables for a parametrized circuit.

Supports simulation with Aer, shot‑based sampling, and the ability to run on real devices.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.opflow import PauliSumOp, StateFn, AerPauliExpectation
from qiskit.providers import BackendV1

class AdvancedFastEstimator:
    """Quantum estimator for parametrised circuits.

    It can evaluate expectation values of arbitrary Pauli or observable operators.
    Supports both ideal statevector simulation and shot‑based sampling.
    """

    def __init__(self,
                 circuit: QuantumCircuit,
                 *,
                 backend: Optional[BackendV1] = None,
                 shots: Optional[int] = None):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.backend = backend or Aer.get_backend("statevector_simulator")
        self.shots = shots

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self,
                 observables: Iterable[PauliSumOp],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables : iterable of PauliSumOp
            Observables expressed as qiskit.opflow operators.
        parameter_sets : sequence of sequences
            Each inner sequence holds the parameters for a single evaluation.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            circ = self._bind(values)
            exp_vals = []
            for op in observables:
                if self.shots is None:
                    # Ideal expectation via statevector
                    sv = Statevector.from_instruction(circ)
                    exp = sv.expectation_value(op)
                    exp_vals.append(exp)
                else:
                    # Sampling expectation via Aer Pauli expectation
                    bound = StateFn(op, coeff=1.0) @ StateFn(circ)
                    exp_val = AerPauliExpectation().convert(bound)
                    job = execute([circ], self.backend, shots=self.shots)
                    result = job.result()
                    # Compute expectation from counts
                    counts = result.get_counts()
                    # Convert counts to expectation value
                    exp = 0.0
                    for bitstring, cnt in counts.items():
                        parity = (-1) ** sum(int(bit) for bit in bitstring)
                        exp += parity * cnt
                    exp = exp / self.shots
                    exp_vals.append(exp)
            results.append(exp_vals)
        return results

__all__ = ["AdvancedFastEstimator"]
