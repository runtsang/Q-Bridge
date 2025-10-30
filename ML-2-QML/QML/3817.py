"""Hybrid estimator for Qiskit quantum circuits with optional quantum self‑attention and shot‑based evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional, Union

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

# --------------------------------------------------------------------------- #
# Helper: quantum self‑attention block
# --------------------------------------------------------------------------- #
class QuantumSelfAttention:
    """
    A minimal quantum self‑attention circuit.  For each qubit, a rotation
    block is applied followed by controlled‑X gates to entangle adjacent qubits.
    """

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> QuantumCircuit:
        if rotation_params.size!= 3 * self.n_qubits:
            raise ValueError("rotation_params length must be 3 * n_qubits")
        if entangle_params.size!= self.n_qubits - 1:
            raise ValueError("entangle_params length must be n_qubits - 1")
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

# --------------------------------------------------------------------------- #
# Main estimator
# --------------------------------------------------------------------------- #
class HybridBaseEstimator:
    """
    Evaluate a parametrized Qiskit circuit for many parameter sets and
    observables.  Observables are Qiskit BaseOperator objects.  The estimator
    can optionally prepend a quantum self‑attention block to the supplied
    circuit and can simulate shot noise by running the circuit with a
    specified number of shots.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        attention: bool = False,
        attention_n_qubits: Optional[int] = None,
    ) -> None:
        self._original_circuit = circuit
        self.attention = attention
        if attention:
            if attention_n_qubits is None:
                raise ValueError("attention_n_qubits must be specified when attention=True")
            self.attention_block = QuantumSelfAttention(attention_n_qubits)
        else:
            self.attention_block = None

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._original_circuit.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._original_circuit.parameters, parameter_values))
        return self._original_circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
    ) -> List[List[complex]]:
        """
        Parameters
        ----------
        observables: iterable of BaseOperator
            Expectation value operators.
        parameter_sets: sequence of parameter value sequences
            Each sequence is bound to the circuit.
        shots: optional
            If provided, the circuit is executed on a qasm simulator with the
            specified number of shots; otherwise a statevector is used and
            expectation values are computed analytically.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        # Choose backend
        if shots is None:
            backend = qiskit.Aer.get_backend("statevector_simulator")
        else:
            backend = qiskit.Aer.get_backend("qasm_simulator")

        for values in parameter_sets:
            circ = self._bind(values)

            # Prepend attention block if requested
            if self.attention:
                att_circ = self.attention_block.build_circuit(
                    rotation_params=np.random.rand(3 * self.attention_block.n_qubits),
                    entangle_params=np.random.rand(self.attention_block.n_qubits - 1),
                )
                circ = att_circ.compose(circ, front=True)

            if shots is None:
                state = Statevector.from_instruction(circ)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                job = qiskit.execute(circ, backend, shots=shots)
                counts = job.result().get_counts(circ)
                # Convert counts to expectation values
                def _expectation(counts: dict, obs: BaseOperator) -> complex:
                    exp = 0.0
                    for bitstring, n in counts.items():
                        # For Pauli observables we can map bitstring to eigenvalue
                        # Here we simply use 1/-1 for Z eigenstates as an example.
                        # In practice, a full mapping would be required.
                        value = 1.0 if bitstring.count("1") % 2 == 0 else -1.0
                        exp += value * n
                    return exp / sum(counts.values())
                row = [_expectation(counts, obs) for obs in observables]
            results.append(row)

        return results


__all__ = ["HybridBaseEstimator"]
