from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

import numpy as np
from qiskit import Aer
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.providers import Backend
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastHybridEstimator:
    """
    Quantum estimator that evaluates expectation values of a parametrized
    circuit and optionally adds finite‑shot noise.  It mimics the behaviour
    of the classical FastBaseEstimator while leveraging Qiskit's state‑vector
    and QASM simulators.

    The class can be instantiated with any ``QuantumCircuit`` and an optional
    backend.  Observables are Qiskit ``BaseOperator`` instances.  When
    ``shots`` is set to a positive integer, the estimator runs a QASM
    simulation and returns the noisy expectation values; otherwise a
    deterministic state‑vector evaluation is performed.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: Backend | None = None,
        shots: int = 0,
        noise_model: object | None = None,
    ) -> None:
        self.circuit = circuit
        self.parameters = list(circuit.parameters)
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.noise_model = noise_model

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables:
            Iterable of Pauli operators or other ``BaseOperator`` instances.
        parameter_sets:
            Sequence of parameter sequences for the circuit.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            bound_circuit = self._bind(values)

            if self.shots == 0:
                # Deterministic state‑vector evaluation
                state = Statevector.from_instruction(bound_circuit)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                # QASM simulation with finite shots
                job = self.backend.run(
                    bound_circuit,
                    shots=self.shots,
                    noise_model=self.noise_model,
                )
                result = job.result()
                counts = result.get_counts(bound_circuit)
                probs = np.array(list(counts.values())) / self.shots
                bits = np.array([int(k, 2) for k in counts.keys()])

                row: List[complex] = []
                for obs in observables:
                    if hasattr(obs, "pauli"):
                        # Assume diagonal Pauli in computational basis
                        eigen = np.array([1 if bit == 0 else -1 for bit in bits])
                        exp = np.sum(eigen * probs)
                        row.append(complex(exp))
                    else:
                        # Fall back to state‑vector expectation
                        state = Statevector.from_instruction(bound_circuit)
                        row.append(state.expectation_value(obs))
                results.append(row)
                continue

            results.append(row)

        return results

    @classmethod
    def from_fcl(cls, n_qubits: int = 1, shots: int = 1024) -> "FastHybridEstimator":
        """
        Convenience constructor that builds a simple fully‑connected‑layer
        quantum circuit used in the FCL example.
        """
        theta = Parameter("theta")
        qc = QuantumCircuit(n_qubits)
        qc.h(range(n_qubits))
        qc.ry(theta, range(n_qubits))
        qc.measure_all()
        return cls(qc, shots=shots)

__all__ = ["FastHybridEstimator"]
