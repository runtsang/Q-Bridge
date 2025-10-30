# SelfAttention_qml.py

"""Quantum circuit that implements a self‑attention style block.

The module follows the quantum‑classifier construction pattern:
* input encoding via RX gates
* a variational rotation block
* entangling CRX gates
* measurement of all qubits

An `evaluate` method based on the FastBaseEstimator pattern
computes expectation values of arbitrary Pauli observables for
a list of parameter sets.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, Sequence, List, Dict

import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class SelfAttentionHybrid:
    """Parametrised quantum self‑attention block.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    backend : qiskit.providers.Provider, optional
        Execution backend; defaults to Aer QASM simulator.
    shots : int, optional
        Number of shots for the measurement.
    """

    def __init__(self, n_qubits: int, backend=None, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------
    def _build_circuit(
        self,
        rotation_params: np.ndarray | None = None,
        entangle_params: np.ndarray | None = None,
    ) -> QuantumCircuit:
        """
        Build a self‑attention circuit with optional parameter binding.

        * `rotation_params` – array of shape ``(n_qubits, 3)``.
          Each row contains the RX, RY, RZ angles.
        * `entangle_params` – array of length ``n_qubits-1``.
          Each element is the angle for a CRX gate.
        """
        circ = QuantumCircuit(self.n_qubits)

        # 1. Input encoding – RX only for simplicity
        encoding = ParameterVector("enc", self.n_qubits)
        for i in range(self.n_qubits):
            circ.rx(encoding[i], i)

        # 2. Variational rotations
        if rotation_params is not None:
            for i in range(self.n_qubits):
                circ.rx(rotation_params[3 * i], i)
                circ.ry(rotation_params[3 * i + 1], i)
                circ.rz(rotation_params[3 * i + 2], i)

        # 3. Entanglement
        if entangle_params is not None:
            for i in range(self.n_qubits - 1):
                circ.crx(entangle_params[i], i, i + 1)

        # 4. Measurement
        circ.measure_all()
        return circ

    # ------------------------------------------------------------------
    # Execution utilities
    # ------------------------------------------------------------------
    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> Dict[str, int]:
        """
        Execute the circuit for a single set of parameters and return
        the raw measurement counts.
        """
        circ = self._build_circuit(rotation_params, entangle_params)
        job = execute(circ, self.backend, shots=self.shots)
        return job.result().get_counts(circ)

    # ------------------------------------------------------------------
    # Fast estimation of expectation values
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Compute expectation values for multiple parameter sets.

        Each parameter set must contain
        ``n_qubits * 3 + (n_qubits - 1)`` floats: first the
        rotation angles, then the entanglement angles.
        """
        results: List[List[complex]] = []
        for params in parameter_sets:
            if len(params)!= self.n_qubits * 3 + (self.n_qubits - 1):
                raise ValueError(
                    f"Expected {self.n_qubits * 3 + (self.n_qubits - 1)} "
                    f"parameters, got {len(params)}."
                )
            rot = np.array(params[: self.n_qubits * 3])
            ent = np.array(params[self.n_qubits * 3 :])
            circ = self._build_circuit(rot, ent)
            state = Statevector.from_instruction(circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results
