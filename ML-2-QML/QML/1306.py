"""Quantum classifier with a parameter‑shift trainable ansatz."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.providers.aer import AerSimulator
from qiskit.opflow import PauliExpectation, StateFn, CircuitStateFn, AerPauliExpectation


class QuantumClassifierModel:
    """
    Variational quantum circuit that emulates the classical interface.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / features.
    depth : int
        Depth of the ansatz.
    backend : str | AerSimulator, optional
        Backend used for simulation. Defaults to AerSimulator('statevector').
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int,
        backend: str | AerSimulator | None = None,
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = (
            backend
            if isinstance(backend, AerSimulator)
            else AerSimulator(backend=backend or "statevector")
        )
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()

    def _build_circuit(self) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)
        for q, param in enumerate(encoding):
            qc.rx(param, q)

        idx = 0
        for _ in range(self.depth):
            for q in range(self.num_qubits):
                qc.ry(weights[idx], q)
                idx += 1
            for q in range(self.num_qubits - 1):
                qc.cz(q, q + 1)

        observables = [
            SparsePauliOp.from_list([(f"I" * i + "Z" + "I" * (self.num_qubits - i - 1), 1.0)])
            for i in range(self.num_qubits)
        ]
        return qc, list(encoding), list(weights), observables

    def get_circuit(self) -> QuantumCircuit:
        """Return the underlying quantum circuit."""
        return self.circuit

    def get_observables(self) -> List[SparsePauliOp]:
        """Return the list of Pauli observables to be measured."""
        return self.observables

    def expectation_values(self, statevector: Statevector | np.ndarray) -> np.ndarray:
        """
        Compute expectation values of the observables for a given statevector.

        Parameters
        ----------
        statevector : Statevector or ndarray
            Statevector to evaluate.
        """
        if isinstance(statevector, np.ndarray):
            statevector = Statevector(statevector)
        exp_vals = []
        for obs in self.observables:
            op = CircuitStateFn(self.circuit, coeff=1.0) @ StateFn(statevector)
            expectation = PauliExpectation().convert(op @ obs)
            exp_vals.append(expectation.eval().real)
        return np.array(exp_vals)

    def parameter_shift_gradient(
        self, statevector: Statevector | np.ndarray
    ) -> np.ndarray:
        """
        Compute gradients w.r.t. all variational parameters using the
        parameter‑shift rule. The gradient is returned as a flat array.
        """
        if isinstance(statevector, np.ndarray):
            statevector = Statevector(statevector)

        grad = np.zeros(len(self.weights))
        shift = np.pi / 2

        for i, param in enumerate(self.weights):
            for sign in [+1, -1]:
                shifted = list(self.weights)
                shifted[i] = param + sign * shift
                # Build a temporary circuit with the shifted parameter
                qc_shifted = self.circuit.copy()
                for idx, p in enumerate(shifted):
                    qc_shifted.assign_parameters({self.weights[idx]: p}, inplace=True)
                sv_shifted = Statevector.from_instruction(qc_shifted)
                exp = self.expectation_values(sv_shifted).sum()
                grad[i] += sign * exp
        grad /= 2.0
        return grad

    @classmethod
    def build_classifier_circuit(
        cls, num_qubits: int, depth: int
    ) -> Tuple["QuantumClassifierModel", List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        """
        Factory that mirrors the classical signature and returns an instance
        together with the encoding, weights and observables.
        """
        model = cls(num_qubits, depth)
        return model, model.encoding, model.weights, model.observables


__all__ = ["QuantumClassifierModel"]
