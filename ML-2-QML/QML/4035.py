"""
Hybrid quantum estimator that extends the original FastBaseEstimator with
optional classical self‑attention observables.  The circuit is evaluated
on a backend and the resulting statevector is used to compute both
expectation values and self‑attention outputs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


# --------------------------------------------------------------------------- #
# Classical self‑attention helper (identical to the ML side)
# --------------------------------------------------------------------------- #
def ClassicalSelfAttention(embed_dim: int = 4) -> callable:
    """
    Stateless self‑attention function that can be used as an additional
    observable after a quantum state has been prepared.
    """
    def _self_attention(inputs: np.ndarray) -> np.ndarray:
        inputs_t = np.asarray(inputs, dtype=np.float32)
        rot = np.random.randn(inputs.shape[1], embed_dim)
        ent = np.random.randn(inputs.shape[1], embed_dim)

        query = inputs_t @ rot
        key   = inputs_t @ ent
        value = inputs_t

        scores = np.exp(query @ key.T / np.sqrt(embed_dim))
        scores /= scores.sum(axis=-1, keepdims=True)
        return scores @ value
    return _self_attention


# --------------------------------------------------------------------------- #
# Hybrid quantum estimator
# --------------------------------------------------------------------------- #
class HybridQuantumEstimator:
    """
    Evaluate a Qiskit circuit with optional classical self‑attention
    observables.  The API mirrors the original FastBaseEstimator and
    exposes an :meth:`evaluate` method that returns both expectation
    values and self‑attention outputs.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parameterised circuit that will be executed on a backend.
    attention : callable | None
        A self‑attention function applied to the parameter vector.  The
        resulting array is returned as an additional observable value.
    backend : qiskit.providers.basebackend.BaseBackend
        The backend used to execute the circuit.  If ``None`` a local
        simulator is used.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        attention: Optional[callable] = None,
        backend=None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.attention = attention
        self.backend = backend

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
        Compute expectation values for every parameter set and observable.
        If a self‑attention function is supplied, its output is appended to
        each row as an additional observable.

        Parameters
        ----------
        observables : iterable
            Quantum operators whose expectation values are desired.
        parameter_sets : sequence
            Each element is a list of parameters matching the circuit.

        Returns
        -------
        List[List[complex]]
            Each row contains the expectation values followed by the
            self‑attention output (if provided).
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for params in parameter_sets:
            state = Statevector.from_instruction(self._bind(params))
            row = [state.expectation_value(obs) for obs in observables]

            if self.attention is not None:
                attn_output = self.attention(np.array(params))
                # The attention output may be multi‑dimensional; we flatten
                # into scalars by taking the mean value.
                row.append(np.mean(attn_output))

            results.append(row)

        return results


__all__ = ["HybridQuantumEstimator"]
