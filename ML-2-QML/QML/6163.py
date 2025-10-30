"""Hybrid estimator for quantum circuits.

This module implements :class:`FastHybridEstimator` that wraps a
``qiskit.circuit.QuantumCircuit``.  It evaluates expectation values of
``BaseOperator`` objects for a batch of parameter sets.  Optional shot
noise is provided by the backend simulator.  A quantum convolution
filter is also available via the ``Conv`` factory from the quantum seed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Sequence, Iterable, Union

import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

# --------------------------------------------------------------------------- #
# Quantum convolution filter (quanvolution)
# --------------------------------------------------------------------------- #

try:
    # Import the quantum Conv factory from the quantum seed.
    from.Conv import Conv as QuantumConv  # type: ignore
except Exception:  # pragma: no cover
    # Minimal fallback using a simple parameterised circuit.
    import qiskit
    from qiskit.circuit.random import random_circuit

    def QuantumConv(kernel_size: int = 2, threshold: float = 0.0) -> QuantumCircuit:
        n_qubits = kernel_size ** 2
        qc = QuantumCircuit(n_qubits)
        theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            qc.rx(theta[i], i)
        qc.barrier()
        qc += random_circuit(n_qubits, 2)
        qc.measure_all()
        return qc


# --------------------------------------------------------------------------- #
# Core estimator
# --------------------------------------------------------------------------- #

class FastHybridEstimator:
    """
    Evaluate a Qiskit circuit for a batch of parameter sets and observables.

    Parameters
    ----------
    circuit : QuantumCircuit
        The parametrised circuit to evaluate.  All parameters must be
        bound before execution.
    """

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    # --------------------------------------------------------------------- #
    # Private helpers
    # --------------------------------------------------------------------- #

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        """Return a circuit with parameters bound to the supplied values."""
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        backend: qiskit.providers.BaseBackend | None = None,
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set.

        Parameters
        ----------
        observables : iterable of BaseOperator
            Operators whose expectation values are desired.
        parameter_sets : sequence of sequences
            Each inner sequence contains the parameters for one evaluation.
        shots : int, optional
            Number of shots for the simulator; if ``None`` the simulator
            uses infinite‑shot (statevector) mode.
        backend : qiskit.providers.BaseBackend, optional
            Backend to run the circuit on.  If ``None`` the Aer qasm
            simulator is used.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        # Choose backend
        if backend is None:
            backend = qiskit.Aer.get_backend("qasm_simulator")

        for values in parameter_sets:
            bound_circ = self._bind(values)

            if shots is None:
                # Statevector simulation for infinite shots
                state = Statevector.from_instruction(bound_circ)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                # Shot‑based simulation
                job = qiskit.execute(
                    bound_circ,
                    backend,
                    shots=shots,
                )
                result = job.result()
                counts = result.get_counts(bound_circ)

                # Compute expectation value as average over all shots
                def expectation(obs: BaseOperator) -> complex:
                    exp = 0.0
                    for bitstring, freq in counts.items():
                        # Convert bitstring to eigenvalue (+1/-1) for each qubit
                        eigen = np.array([1 if b == "1" else -1 for b in bitstring[::-1]])
                        exp += np.real(obs.data @ eigen) * freq
                    return exp / (shots * len(counts))

                row = [expectation(obs) for obs in observables]

            results.append(row)

        return results


__all__ = ["FastHybridEstimator"]
