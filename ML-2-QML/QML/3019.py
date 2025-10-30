"""Quantum estimator that uses a variational circuit with 4 qubits.

The circuit accepts 4 input angles and 4 trainable weight angles,
entangles the qubits, and measures the sum of Pauli‑Z observables.
The module can be used with either a simulator or a real device
through Qiskit’s StatevectorEstimator primitive.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN


class EstimatorQNN:
    """Quantum estimator that wraps a qiskit EstimatorQNN."""

    def __init__(self) -> None:
        # 4 input parameters (angles) and 4 trainable weight parameters
        self.input_params = [Parameter(f"x{i}") for i in range(4)]
        self.weight_params = [Parameter(f"w{i}") for i in range(4)]
        self.params = self.input_params + self.weight_params

        # Build a 4‑qubit variational circuit
        self.circuit = QuantumCircuit(4)
        for i in range(4):
            self.circuit.ry(self.input_params[i], i)
            self.circuit.rx(self.weight_params[i], i)
        # Entangle neighbouring qubits
        for i in range(3):
            self.circuit.cx(i, i + 1)
        self.circuit.barrier()

        # Observable: sum of Pauli‑Z on each qubit
        self.observable = SparsePauliOp.from_list([("Z" * 4, 1)])

        # Estimator primitive and wrapper
        self.estimator = StatevectorEstimator()
        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def __call__(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Evaluate the quantum circuit for a batch of inputs and weight vectors.

        Parameters
        ----------
        inputs : np.ndarray
            Array of shape (4,) containing input angles.
        weights : np.ndarray
            Array of shape (4,) containing trainable weight angles.

        Returns
        -------
        np.ndarray
            Array of expectation values (one per input batch).
        """
        if inputs.shape!= (4,) or weights.shape!= (4,):
            raise ValueError("inputs and weights must each be 1‑D arrays of length 4.")
        param_dict = {p: v for p, v in zip(self.params, np.concatenate([inputs, weights]))}
        # Evaluate
        results = self.estimator_qnn.evaluate([param_dict])
        return np.array([res.values[0] for res in results])


__all__ = ["EstimatorQNN"]
