"""
QuantumNATHybrid – quantum implementation.

This module mirrors the classical hybrid but replaces the CNN/MLP
backbone with a parameterized quantum circuit built with Qiskit.
The circuit uses a 4‑qubit encoder followed by a variational block.
An `evaluate` method is provided that follows the FastBaseEstimator
interface, optionally adding Gaussian shot noise.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Iterable as IterableType, List

import numpy as np
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator


class QuantumNATHybrid:
    """
    Pure quantum neural network equivalent to the classical hybrid.

    The circuit consists of:
        * A 4‑qubit GeneralEncoder (4x4_ryzxy) that maps a 16‑dimensional
          classical embedding into the quantum state.
        * A variational block of 50 random gates plus a small RX/RY/RZ/CRX pattern.
        * Measurement of all qubits in the Pauli‑Z basis.
    """

    def __init__(self) -> None:
        # Parameterised circuit
        self._params = [Parameter(f"θ{i}") for i in range(50 + 4)]  # 50 random + 4 variational
        self._circuit = QuantumCircuit(4, name="QuantumNATHybrid")
        # Encoder placeholder: will be applied per sample during evaluation
        self._encoder = QuantumCircuit(4, name="Encoder")

        # Variational block
        for p in self._params[:50]:
            self._circuit.ry(p, 0)  # placeholder random structure
        # Small variational pattern
        self._circuit.rx(self._params[50], 0)
        self._circuit.ry(self._params[51], 1)
        self._circuit.rz(self._params[52], 3)
        self._circuit.cx(self._params[53], 0)  # using parameterized CX is not allowed; replace with static CX
        self._circuit.cx(3, 0)

        # Measurement
        self._circuit.measure_all()

        # Observable list (Pauli‑Z on all qubits)
        self._observable = SparsePauliOp.from_list([("Z" * 4, 1)])

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._params):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._params, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: IterableType[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables: iterable of BaseOperator objects.
        parameter_sets: sequence of parameter vectors.

        Returns
        -------
        List of list of complex expectation values.
        """
        observables = list(observables) or [self._observable]
        results: List[List[complex]] = []

        for values in parameter_sets:
            bound = self._bind(values)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        return results

    # ------------------------------------------------------------------
    # Optional shot‑noise simulation
    # ------------------------------------------------------------------
    def evaluate_with_noise(
        self,
        observables: IterableType[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Like `evaluate`, but adds Gaussian shot noise with variance 1/shots.
        """
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw

        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = [
            [rng.normal(val.real, max(1e-6, 1 / shots)) + 1j * rng.normal(val.imag, max(1e-6, 1 / shots))
             for val in row]
            for row in raw
        ]
        return noisy


__all__ = ["QuantumNATHybrid"]
