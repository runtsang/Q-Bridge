"""Quantum self‑attention circuit with expectation‑value estimator.

The module defines a `SelfAttention` class that builds a parameterised
circuit from rotation and entanglement angles.  `FastEstimator` evaluates
expectation values for a list of observables and can add Gaussian
shot noise to emulate finite‑sample statistics.  The run method
returns the noisy or noiseless expectation values.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, Sequence, List, Union

class FastEstimator:
    """Evaluate expectation values for a parameterised circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        The quantum circuit with symbolic parameters.
    shots : int | None, optional
        If provided, Gaussian shot noise with variance ``1/shots`` is
        added to each expectation value to mimic finite‑sample estimation.
    seed : int | None, optional
        Random seed for reproducible noise.
    """
    def __init__(self, circuit: QuantumCircuit, shots: int | None = None, seed: int | None = None):
        self.circuit = circuit
        self.shots = shots
        self.seed = seed
        self._params = list(circuit.parameters)

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= len(self._params):
            raise ValueError("Mismatch between supplied parameters and circuit parameters.")
        mapping = dict(zip(self._params, param_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if self.shots is not None:
            rng = np.random.default_rng(self.seed)
            noisy = []
            for row in results:
                noisy.append(
                    [complex(
                        rng.normal(row[i].real, 1 / np.sqrt(self.shots)).real,
                        rng.normal(row[i].imag, 1 / np.sqrt(self.shots)).imag
                    ) for i in range(len(row))]
                )
            return noisy
        return results


class SelfAttention:
    """Quantum self‑attention block built from rotation and entanglement parameters.

    Parameters
    ----------
    n_qubits : int, default 4
        Number of qubits in the block.  One rotation triplet per qubit
        and one entanglement angle per adjacent qubit pair are expected.
    """
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr, self.cr)
        # Apply rotations to each qubit
        for i in range(self.n_qubits):
            r = rotation_params[3 * i : 3 * i + 3]
            qc.rx(r[0], i)
            qc.ry(r[1], i)
            qc.rz(r[2], i)
        # Entangle adjacent qubits with controlled‑rotation gates
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)
        qc.barrier()
        return qc

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        observables: Iterable[BaseOperator],
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Return expectation values of the given observables.

        The circuit is built from the supplied parameters, bound, and
        evaluated by `FastEstimator`.  If ``shots`` is given, shot noise
        is added to each expectation value.
        """
        circ = self._build_circuit(rotation_params, entangle_params)
        estimator = FastEstimator(circ, shots=shots, seed=seed)
        # Concatenate parameters for binding
        param_set = list(rotation_params) + list(entangle_params)
        return estimator.evaluate(observables, [param_set])


__all__ = ["SelfAttention", "FastEstimator"]
