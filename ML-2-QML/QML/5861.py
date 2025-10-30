"""Hybrid estimator combining Qiskit variational circuits and quantum self‑attention."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Union

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

class QuantumSelfAttention:
    """Self‑attention style subcircuit that can be embedded in a larger circuit."""

    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_subcircuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        return circuit

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ):
        circuit = self._build_subcircuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)


class FastHybridEstimator:
    """
    Evaluate either a Qiskit circuit or a quantum self‑attention subcircuit.
    The constructor accepts a QuantumCircuit or a QuantumSelfAttention
    instance.  The API mirrors the original FastBaseEstimator but adds
    support for variational parameters and shot noise.
    """

    def __init__(self, circuit: Union[QuantumCircuit, QuantumSelfAttention]) -> None:
        self.circuit = circuit
        self._is_subcircuit = isinstance(circuit, QuantumSelfAttention)
        if self._is_subcircuit:
            # For subcircuit we need a backend; default to qasm_simulator
            self.backend = qiskit.Aer.get_backend("qasm_simulator")
        else:
            self.backend = None

    def evaluate(
        self,
        observables: Iterable[BaseOperator] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        For a full circuit: compute expectation values of the supplied
        observables.  For a self‑attention subcircuit: run the circuit
        and return measurement counts.
        """
        if parameter_sets is None:
            raise ValueError("parameter_sets must be provided")

        if self._is_subcircuit:
            # Subcircuit path
            results: List[List[complex]] = []
            for params in parameter_sets:
                rot_len = len(params) // 3
                rot_params = np.array(params[:rot_len])
                ent_params = np.array(params[rot_len:2 * rot_len])
                counts = self.circuit.run(
                    self.backend, rot_params, ent_params, shots=shots or 1024
                )
                results.append([counts])
            return results

        # Full circuit path
        observables = list(observables) if observables is not None else [Statevector.zero(self.circuit.num_qubits)]
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound_circ = self.circuit.assign_parameters(dict(zip(self.circuit.parameters, values)), inplace=False)
            state = Statevector.from_instruction(bound_circ)
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [complex(rng.normal(float(val.real), max(1e-6, 1 / shots)),
                                 rng.normal(float(val.imag), max(1e-6, 1 / shots))) for val in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastHybridEstimator", "QuantumSelfAttention"]
