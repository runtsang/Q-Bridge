"""Hybrid quantum circuit that mirrors the classical structure of HybridFullyConnectedLayer."""

import numpy as np
from qiskit import Aer
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator


class HybridFullyConnectedLayer:
    """
    Quantum implementation of a hybrid fully connected layer.

    The circuit encodes two parameters:
    - input_param: drives an Ry rotation,
    - weight_param: drives an Rx rotation.

    The observable is the Pauli‑Y operator, and the expectation value
    is returned as the layer output.

    This design combines the FCL circuit (single‑qubit H, Ry, measurement)
    with the EstimatorQNN architecture (parameterised circuit, observable,
    StatevectorEstimator). It thus provides a quantum counterpart to
    the classical HybridFullyConnectedLayer.
    """

    def __init__(self, backend=None, shots: int = 1024):
        self.n_qubits = 1
        self.backend = backend or Aer.get_backend("statevector_simulator")
        self.shots = shots

        # Define parameters
        self.input_param = Parameter("input")
        self.weight_param = Parameter("weight")

        # Build circuit
        self.circuit = QuantumCircuit(self.n_qubits)
        self.circuit.h(0)
        self.circuit.ry(self.input_param, 0)
        self.circuit.rx(self.weight_param, 0)

        # Observable Y
        self.observable = SparsePauliOp.from_list([("Y", 1)])

        # Estimator
        self.estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=[self.input_param],
            weight_params=[self.weight_param],
            estimator=self.estimator,
        )

    def run(self, input_val: float, weight_val: float) -> np.ndarray:
        """
        Evaluate the quantum circuit for given input and weight values.

        Parameters
        ----------
        input_val : float
            Value to bind to the input parameter.
        weight_val : float
            Value to bind to the weight parameter.

        Returns
        -------
        np.ndarray
            Quantum expectation value as a single‑element array.
        """
        result = self.estimator_qnn.predict(
            input_vals=[input_val],
            weight_vals=[weight_val],
        )
        return np.array([result])


__all__ = ["HybridFullyConnectedLayer"]
