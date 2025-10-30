"""Hybrid quantum self‑attention that evaluates attention scores via
EstimatorQNN.  The circuit is parameterised by rotation and entanglement
angles; the EstimatorQNN is used to obtain expectation values of the Y
observable, which serve as attention weights.

The module mirrors the seed SelfAttention quantum implementation and
extends it with the EstimatorQNN example to provide a quantum
regression of the attention logits.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator


class HybridQuantumAttention:
    """Quantum self‑attention block using EstimatorQNN to compute
    attention logits from a parameterised circuit.
    """

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        # Parameterised rotation and entanglement angles
        self.rotation_params = [Parameter(f"r_{i}") for i in range(n_qubits * 3)]
        self.entangle_params = [Parameter(f"e_{i}") for i in range(n_qubits - 1)]
        self.circuit = self._build_param_circuit()

    def _build_param_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(self.rotation_params[3 * i], i)
            qc.ry(self.rotation_params[3 * i + 1], i)
            qc.rz(self.rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(self.entangle_params[i], i, i + 1)
        return qc

    def run(
        self,
        rotation_values: np.ndarray,
        entangle_values: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the circuit for the supplied parameter values and return
        the expectation value of the Y observable on each qubit.  The
        returned array has shape ``(n_qubits,)`` and can be interpreted
        as attention logits for the corresponding input positions.

        Parameters
        ----------
        rotation_values : np.ndarray
            Array of length ``3 * n_qubits`` containing the rotation angles.
        entangle_values : np.ndarray
            Array of length ``n_qubits - 1`` containing the CRX angles.
        shots : int, optional
            Number of shots for the simulation (default 1024).

        Returns
        -------
        np.ndarray
            Expectation values of the Y observable for each qubit.
        """
        # Build EstimatorQNN instance
        observable = SparsePauliOp.from_list([("Y" * self.n_qubits, 1.0)])
        estimator = Estimator()
        estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=[observable],
            input_params=self.rotation_params,
            weight_params=self.entangle_params,
            estimator=estimator,
        )

        # Map parameters to values
        param_map = {
            p: v for p, v in zip(self.rotation_params, rotation_values)
        }
        param_map.update(
            {p: v for p, v in zip(self.entangle_params, entangle_values)}
        )

        # Evaluate expectation values
        expectation = estimator_qnn.evaluate(**param_map)
        # estimator_qnn.evaluate returns a tuple of arrays; we flatten it
        return np.array(expectation).reshape(-1)
