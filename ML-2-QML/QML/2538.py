"""FastBaseEstimator extended for quantum circuits with optional quanvolution filtering."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.pauli import PauliZ


class QuanvolutionQuantumFilter:
    """Random two‑qubit kernel applied to 2×2 image patches using a Qiskit circuit."""
    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self.n_wires = 4
        rng = np.random.default_rng(seed)
        self.circuit = QuantumCircuit(self.n_wires)
        # Random single‑qubit rotations
        for i in range(self.n_wires):
            self.circuit.ry(rng.uniform(0, 2*np.pi), i)
        # Small entangling layer
        self.circuit.cx(0, 1)
        self.circuit.cx(2, 3)
        # Additional rotations
        for i in range(self.n_wires):
            self.circuit.ry(rng.uniform(0, 2*np.pi), i)

    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Apply the filter to a batch of 2×2 patches.
        Parameters
        ----------
        data : (batch, 4) array of real numbers.
        Returns
        -------
        (batch, 4) array of expectation values of PauliZ on each qubit.
        """
        batch = data.shape[0]
        meas = np.zeros((batch, self.n_wires))
        for idx, patch in enumerate(data):
            qc = self.circuit.copy()
            for i, val in enumerate(patch):
                qc.ry(val, i)
            state = Statevector.from_instruction(qc)
            for i in range(self.n_wires):
                pauli = PauliZ('I'*i + 'Z' + 'I'*(self.n_wires-i-1))
                meas[idx, i] = state.expectation_value(pauli)
        return meas


class FastBaseEstimator:
    """Estimator that evaluates a parametrised quantum circuit, optionally followed by a quanvolution filter."""
    def __init__(self, circuit: QuantumCircuit, *, use_quanvolution: bool = False, seed: int | None = None) -> None:
        self.circuit = circuit
        self.parameters = list(circuit.parameters)
        self.use_quanvolution = use_quanvolution
        self.qfilter: Optional[QuanvolutionQuantumFilter] = (
            QuanvolutionQuantumFilter(seed) if use_quanvolution else None
        )

    def _bind(self, param_vals: Sequence[float]) -> QuantumCircuit:
        if len(param_vals)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, param_vals))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set and observable.
        If a quanvolution filter is enabled, the statevector is first processed by the filter
        before the expectation value is taken.
        """
        results: List[List[complex]] = []
        for params in parameter_sets:
            circ = self._bind(params)
            state = Statevector.from_instruction(circ)
            if self.use_quanvolution:
                # Toy illustration: apply filter to the first 4 amplitudes of the state
                raw = state.data.real.reshape(1, -1)[:,:self.qfilter.n_wires]
                meas = self.qfilter.apply(raw)
                # Build a new statevector from the filtered measurements
                new_state = Statevector(meas[0], dims=(2,)*self.qfilter.n_wires)
                state = new_state
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


__all__ = ["FastBaseEstimator", "QuanvolutionQuantumFilter"]
