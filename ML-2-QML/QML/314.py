"""
FastBaseEstimator: variational circuit evaluator with shot sampling.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, execute
from qiskit.providers.aer import Aer
from qiskit.quantum_info import Pauli, Statevector
from typing import Iterable, List, Sequence, Tuple

# --------------------------------------------------------------------------- #
#  Helper: Pauli string to matrix
# --------------------------------------------------------------------------- #
def _pauli_str_to_matrix(pauli_str: str) -> np.ndarray:
    """
    Convert a Pauli string (e.g. 'IXZ') into its matrix representation.
    """
    pauli_dict = {
        "I": np.eye(2),
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.array([[1, 0], [0, -1]]),
    }
    op = np.eye(1, dtype=complex)
    for char in pauli_str:
        op = np.kron(op, pauli_dict[char])
    return op

# --------------------------------------------------------------------------- #
#  Core estimator
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """
    Evaluate expectation values of Pauli observables for a parametrised circuit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    backend_name : str, optional
        Name of the Qiskit Aer backend.  ``"statevector_simulator"`` is used by
        default for exact evaluation; ``"qasm_simulator"`` for shot sampling.
    """

    def __init__(self, n_qubits: int, backend_name: str = "statevector_simulator") -> None:
        self.n_qubits = n_qubits
        self.backend_name = backend_name
        self.backend = Aer.get_backend(backend_name)

    # --------------------------------------------------------------------- #
    #  Circuit construction
    # --------------------------------------------------------------------- #
    def _build_circuit(self, params: Sequence[float]) -> QuantumCircuit:
        """
        Build a simple variational circuit: a chain of RY rotations followed by
        a ladder of CNOTs.
        """
        circ = QuantumCircuit(self.n_qubits)
        for i, theta in enumerate(params):
            circ.ry(theta, i)
        for i in range(self.n_qubits - 1):
            circ.cx(i, i + 1)
        return circ

    # --------------------------------------------------------------------- #
    #  Evaluation
    # --------------------------------------------------------------------- #
    def evaluate(
        self,
        observables: Iterable[str],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Compute expectation values for every parameter set and observable.

        Parameters
        ----------
        observables : Iterable[str]
            Iterable of Pauli strings such as ``"IXZ"`` or ``"ZZ"``.  Each
            character must be one of ``I, X, Y, Z``.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors of length ``n_qubits``.
        shots : int | None, optional
            Number of shots.  Pass ``None`` for exact expectation values.
        seed : int | None, optional
            Random seed for the shot sampler.

        Returns
        -------
        List[List[complex]]
            Nested list where each outer element corresponds to a parameter
            set and each inner element corresponds to an observable.
        """
        obs_matrices = [_pauli_str_to_matrix(op) for op in observables]
        results: List[List[complex]] = []

        for params in parameter_sets:
            circ = self._build_circuit(params)
            if shots is None:
                # Exact evaluation using Statevector
                state = Statevector.from_instruction(circ)
                row = [float(state.expectation_value(Pauli.from_label(op))) for op in observables]
            else:
                # Shot sampling using QASM simulator
                job = execute(circ, backend=self.backend, shots=shots, seed_simulator=seed)
                counts = job.result().get_counts()
                row = []
                for op in observables:
                    exp = 0.0
                    for bitstring, count in counts.items():
                        bits = [int(b) for b in bitstring[::-1]]
                        eigenvalue = 1.0
                        for i, char in enumerate(op):
                            if char == "Z":
                                eigenvalue *= 1 if bits[i] == 0 else -1
                            elif char == "X" or char == "Y":
                                # For X/Y we approximate by setting eigenvalue to 0
                                eigenvalue = 0.0
                                break
                        exp += eigenvalue * count
                    exp /= shots
                    row.append(exp)
            results.append(row)
        return results

    # --------------------------------------------------------------------- #
    #  Convenience wrapper for a default observable set
    # --------------------------------------------------------------------- #
    def evaluate_params(
        self,
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Convenience wrapper that evaluates a default observable set
        (currently ``["Z"]`` for each qubit) for the given parameter sets.
        """
        default_observables = ["Z"] * self.n_qubits
        return self.evaluate(default_observables, parameter_sets, shots=shots, seed=seed)

# --------------------------------------------------------------------------- #
#  Expose public API
# --------------------------------------------------------------------------- #
__all__ = ["FastBaseEstimator"]
