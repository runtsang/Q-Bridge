from __future__ import annotations

import numpy as np
from typing import Iterable, Sequence, List, Optional
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class HybridFCL:
    """
    Quantum counterpart of :class:`HybridFCL`.  The circuit consists of a
    single parameterised rotation on each qubit followed by a measurement
    in the computational basis.  The expectation value of the Pauli‑Z
    operator on the first qubit is returned.  The class also implements
    an :py:meth:`evaluate` routine that mirrors the classical estimator
    and accepts arbitrary Pauli‑Z observables.
    """

    def __init__(
        self,
        n_qubits: int = 1,
        backend: Optional[object] = None,
        shots: int = 1024,
    ) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self._circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        theta = qc.parameters[0] if qc.parameters else qc.circuit.Parameter("theta")
        qc.h(range(self.n_qubits))
        qc.barrier()
        qc.ry(theta, range(self.n_qubits))
        qc.measure_all()
        return qc

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit for a list of parameter values and return the
        expectation of the Pauli‑Z operator on qubit 0.
        """
        job = execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self._circuit.parameters[0]: theta} for theta in thetas],
        )
        result = job.result()
        counts = result.get_counts(self._circuit)
        total = sum(counts.values())
        expectation = sum(int(state, 2) * freq for state, freq in counts.items()) / total
        return np.array([expectation])

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set and observable.
        The observable is expected to be a Pauli‑Z or similar operator that
        can be applied to the statevector returned by Qiskit.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            bound = self._circuit.assign_parameters({self._circuit.parameters[0]: values[0]}, inplace=False)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(op) for op in observables]
            results.append(row)

        return results


__all__ = ["HybridFCL"]
