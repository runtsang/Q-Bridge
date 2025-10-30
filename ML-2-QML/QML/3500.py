"""
Hybrid FastBaseEstimator for quantum circuits with optional self‑attention.

This module mirrors the classical implementation but uses Qiskit to
evaluate parametrised circuits.  It adds:
- An optional quantum self‑attention sub‑circuit.
- Support for shot‑based noise and a deterministic mode.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

# Import the quantum self‑attention helper
try:
    from.SelfAttention import SelfAttention as QuantumSelfAttention
except Exception:  # pragma: no cover
    QuantumSelfAttention = None


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrised circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        The base circuit to evaluate.
    self_attention : bool, default False
        If True, a quantum self‑attention block is appended to the circuit.
    attention_n_qubits : int, optional
        The number of qubits for the attention block.  Required if
        ``self_attention`` is True.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        self_attention: bool = False,
        attention_n_qubits: Optional[int] = None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.self_attention = self_attention
        if self_attention:
            if attention_n_qubits is None:
                raise ValueError("attention_n_qubits must be specified when self_attention is True")
            self._attention = QuantumSelfAttention(attention_n_qubits)  # type: ignore
        else:
            self._attention = None
        self._backend = Aer.get_backend("qasm_simulator")

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def _build_with_attention(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        base_circuit: QuantumCircuit,
    ) -> QuantumCircuit:
        """Attach a self‑attention sub‑circuit to the base circuit."""
        attention_circ = self._attention._build_circuit(rotation_params, entangle_params)
        # Combine: first attention, then base
        composite = base_circuit.compose(attention_circ, front=True, inplace=False)
        return composite

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
        attention_params: Optional[tuple[np.ndarray, np.ndarray]] = None,
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set.

        Parameters
        ----------
        observables : Iterable[BaseOperator]
            Operators to measure expectation values of.
        parameter_sets : Sequence[Sequence[float]]
            Parameter vectors for the circuit.
        shots : int, optional
            If provided, the circuit is executed on a shot‑based backend.
        seed : int, optional
            Random seed for reproducible shot noise.
        attention_params : tuple[np.ndarray, np.ndarray], optional
            Rotation and entangle parameters for the attention block.
            Required if ``self_attention`` is True.

        Returns
        -------
        List[List[complex]]
            Expectation values for each parameter set.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        if self.self_attention:
            if attention_params is None:
                raise ValueError("attention_params must be supplied when self_attention is True")
            rot, ent = attention_params
        else:
            rot = ent = None

        for values in parameter_sets:
            base_circ = self._bind(values)
            if self.self_attention:
                base_circ = self._build_with_attention(rot, ent, base_circ)

            if shots is None:
                # Deterministic evaluation via Statevector
                state = Statevector.from_instruction(base_circ)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                job = qiskit.execute(base_circ, self._backend, shots=shots, seed_simulator=seed)
                counts = job.result().get_counts(base_circ)
                # Convert counts to probabilities
                probs = {k: v / shots for k, v in counts.items()}
                # Build a density matrix from measurement outcomes
                density = qiskit.quantum_info.DensityMatrix.from_counts(counts, base_circ.num_qubits)
                row = [density.expectation_value(obs) for obs in observables]
            results.append(row)

        return results


__all__ = ["FastBaseEstimator"]
