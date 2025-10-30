"""HybridEstimatorFCL: Quantum implementation inspired by FCL and EstimatorQNN.

The class builds a variational quantum circuit using Qiskit and the
EstimatorQNN wrapper.  It exposes the same `run` method as the classical
counterpart, enabling side‑by‑side comparison or hybrid training.  The
circuit consists of a single qubit, a Hadamard gate, a parameterised Ry
(rotation around Y) for the input, and an Rx (rotation around X) for the
weight.  The expectation value of the Y Pauli operator is returned.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator
from typing import Iterable


class HybridEstimatorFCL:
    """
    Quantum hybrid estimator.

    Parameters
    ----------
    n_qubits : int, default 1
        Number of qubits in the variational circuit.
    shots : int, default 100
        Number of shots for the backend simulator.
    """

    def __init__(self, n_qubits: int = 1, shots: int = 100) -> None:
        self.n_qubits = n_qubits
        self.shots = shots

        # Define trainable parameters
        self.input_param = Parameter("input")
        self.weight_param = Parameter("weight")

        # Construct a simple variational circuit
        qc = QuantumCircuit(n_qubits)
        qc.h(0)
        qc.ry(self.input_param, 0)
        qc.rx(self.weight_param, 0)
        qc.measure_all()

        # Observable for expectation value
        observable = SparsePauliOp.from_list([("Y" * n_qubits, 1)])

        # Quantum estimator
        estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[self.input_param],
            weight_params=[self.weight_param],
            estimator=estimator,
        )

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the variational circuit and return the expectation value.

        Parameters
        ----------
        thetas : Iterable[float]
            Must contain two values: the input rotation and the weight rotation.

        Returns
        -------
        np.ndarray
            A 1‑D array containing the scalar expectation value.
        """
        if len(thetas) < 2:
            raise ValueError("EstimatorQNN requires two parameters: input and weight.")
        # EstimatorQNN expects a list of inputs followed by weights
        expectation = self.estimator_qnn([thetas[0], thetas[1]])
        return np.array([expectation.squeeze()])

def FCL() -> HybridEstimatorFCL:
    """Return an instance of the quantum hybrid estimator for backward compatibility."""
    return HybridEstimatorFCL()
