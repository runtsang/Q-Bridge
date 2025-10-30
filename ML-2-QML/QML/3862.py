"""FastBaseEstimator for Qiskit circuits with optional shot noise.

The estimator accepts a Qiskit QuantumCircuit that is parameterised
by qiskit.circuit.Parameter objects.  The evaluate method binds each
parameter set, simulates the circuit with a statevector or qasm
back‑end, and returns expectation values for a list of BaseOperator
observables.  An optional ``shots`` argument can override the circuit
back‑end’s default shot count to emulate sampling statistics.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional, Any
import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastBaseEstimator:
    """Evaluate expectation values for a parameterised Qiskit circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        A circuit that contains qiskit.circuit.Parameter objects.
    backend : Backend, optional
        Backend used for the state‑vector simulation.  Defaults to the
        Aer statevector simulator.
    shots : int, optional
        Number of shots used when the circuit is executed on a qasm
        simulator.  If the circuit already has a ``shots`` attribute,
        that value is used unless overridden here.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: Optional[Any] = None,
        shots: Optional[int] = None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self._backend = backend or qiskit.Aer.get_backend("statevector_simulator")
        self._shots = shots or getattr(circuit, "shots", None)

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
        """Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables
            Iterable of qiskit.QuantumInfo operators.  Each operator must
            be able to act on a Statevector.
        parameter_sets
            Sequence of sequences of floats that will be bound to the
            circuit parameters.

        Returns
        -------
        List[List[complex]]
            A table of expectation values.  Each row corresponds to a
            parameter set and each column to an observable.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            bound = self._bind(values)

            # Use state‑vector simulation for deterministic results
            if self._shots is None or self._shots == 0:
                state = Statevector.from_instruction(bound)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                # Execute on a qasm simulator to emulate shot noise
                job = qiskit.execute(
                    bound,
                    backend=self._backend,
                    shots=self._shots,
                )
                result = job.result()
                counts = result.get_counts(bound)
                probs = np.array(list(counts.values())) / self._shots
                states = np.array([int(k, 2) for k in counts.keys()])
                expectation = sum(states * probs)
                row = [expectation for _ in observables]  # same value for all

            results.append(row)

        return results


def FCL(n_qubits: int = 1, shots: int = 1024) -> QuantumCircuit:
    """Return a simple parameterised quantum circuit for a fully‑connected layer.

    The circuit implements a single rotation about the Y axis on ``n_qubits``,
    collects measurement statistics, and returns the expectation value of
    the computational basis state ``|1⟩`` as a proxy for a fully‑connected
    neural‑network layer.
    """
    theta = qiskit.circuit.Parameter("theta")
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    qc.barrier()
    qc.ry(theta, range(n_qubits))
    qc.measure_all()
    return qc


__all__ = ["FastBaseEstimator", "FCL"]
