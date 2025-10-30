"""FastBaseEstimator for quantum circuits with sampling and shot‑noise."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.opflow import PauliSumOp
from qiskit.opflow import PauliOp, StateFn, CircuitStateFn, ExpectationFactory


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrised circuit."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: Optional[object] = None,
        shots: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        circuit : QuantumCircuit
            The parameterised circuit.
        backend : object | None
            Qiskit backend to use.  If ``None`` the Aer statevector simulator
            is used.  If *shots* is supplied a qasm simulator with that number
            of shots is created.
        shots : int | None
            Number of measurement shots.  If supplied a shot‑based simulation
            is performed and expectation values are estimated from the
            measurement distribution.
        """
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

        if backend is None:
            if shots is None:
                self._backend = Aer.get_backend("statevector_simulator")
            else:
                self._backend = Aer.get_backend("qasm_simulator")
                self._shots = shots
        else:
            self._backend = backend
            self._shots = getattr(backend, "shots", shots)

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
        """
        Compute expectation values of *observables* for each parameter set.
        If a shot‑based backend is used, the returned values are the mean
        over the measurement samples.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        # Pre‑convert observables to PauliSumOps for efficient evaluation
        pauli_ops = [PauliSumOp.from_operator(obs) for obs in observables]

        for params in parameter_sets:
            bound_circuit = self._bind(params)

            if isinstance(self._backend, Aer.AerSimulator) and self._backend.name() == "statevector_simulator":
                state = Statevector.from_instruction(bound_circuit)
                row = [state.expectation_value(op).real for op in pauli_ops]
            else:
                # shot‑based simulation
                job = execute(bound_circuit, self._backend, shots=self._shots)
                result = job.result()
                counts = result.get_counts()
                # Build density matrix from counts
                rho = Statevector.from_counts(counts, bound_circuit.num_qubits).data
                state = Statevector(rho)
                row = [state.expectation_value(op).real for op in pauli_ops]
            results.append(row)

        return results


__all__ = ["FastBaseEstimator"]
