"""HybridEstimator – quantum estimator that can embed a quanvolution filter."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Sequence

import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class HybridEstimator:
    """
    Quantum estimator that evaluates expectation values for a parametrized
    circuit.  An optional quanvolution filter can be prepended to the main
    circuit.

    Parameters
    ----------
    circuit:
        The primary circuit used for evaluation.
    conv:
        Optional quanvolution circuit that transforms input data before the
        main circuit.  Each ``parameter_set`` must contain two parts:
        ``conv_params`` and ``circuit_params``.
    """

    def __init__(self, circuit: QuantumCircuit, conv: QuantumCircuit | None = None) -> None:
        self._circuit = circuit
        self._conv = conv
        self._parameters = list(circuit.parameters)
        self._conv_params = list(conv.parameters) if conv else []

    def _bind(self, circuit_params: Sequence[float]) -> QuantumCircuit:
        if len(circuit_params)!= len(self._parameters):
            raise ValueError("Circuit parameter count mismatch.")
        mapping = dict(zip(self._parameters, circuit_params))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def _bind_conv(self, conv_params: Sequence[float]) -> QuantumCircuit:
        if not self._conv:
            return None
        if len(conv_params)!= len(self._conv_params):
            raise ValueError("Conv parameter count mismatch.")
        mapping = dict(zip(self._conv_params, conv_params))
        return self._conv.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set and observable.

        Each element of ``parameter_sets`` must be a flat list containing
        first the conv parameters (if any) followed by the main circuit
        parameters.  The method automatically splits the list.

        Parameters
        ----------
        observables:
            Quantum operators whose expectation values are to be measured.
        parameter_sets:
            Sequence of flat parameter lists.
        """
        results: List[List[complex]] = []
        for pset in parameter_sets:
            # split parameters
            if self._conv:
                conv_p, circ_p = pset[: len(self._conv_params)], pset[len(self._conv_params) :]
                conv_circ = self._bind_conv(conv_p)
                main_circ = self._bind(circ_p)
                full_circ = qiskit.circuit.QuantumCircuit(
                    main_circ.num_qubits + conv_circ.num_qubits
                )
                full_circ.compose(conv_circ, inplace=True)
                full_circ.compose(main_circ, inplace=True)
                state = Statevector.from_instruction(full_circ)
            else:
                main_circ = self._bind(pset)
                state = Statevector.from_instruction(main_circ)

            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


__all__ = ["HybridEstimator"]
