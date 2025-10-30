import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator

class HybridFCL:
    """
    Quantum implementation mirroring the classical HybridFCL.
    The circuit is a single qubit with H, Ry(input1), Rx(weight1) gates.
    The weight is treated as a trainable parameter (weight1) and the
    input as an external parameter (input1). The observable is Y,
    so the expectation value equals sin(weight1 + input1).
    The EstimatorQNN wrapper provides a convenient interface for
    evaluating the expectation on a simulator.
    """
    def __init__(self, backend=None, shots=1024):
        # Parameters
        self.input1 = Parameter("input1")
        self.weight1 = Parameter("weight1")

        # Build circuit
        self.circuit = QuantumCircuit(1)
        self.circuit.h(0)
        self.circuit.ry(self.input1, 0)
        self.circuit.rx(self.weight1, 0)
        self.circuit.measure_all()

        # Observable
        observable = SparsePauliOp.from_list([("Y", 1)])

        # Estimator
        estimator = StatevectorEstimator(backend=backend)
        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=observable,
            input_params=[self.input1],
            weight_params=[self.weight1],
            estimator=ester
        )

    def run(self, inputs: np.ndarray, weight: float) -> np.ndarray:
        """
        Evaluate the circuit for given classical inputs and weight.

        Args:
            inputs (np.ndarray): shape (..., 2) – first column is input1,
                                 second column could be ignored or used for
                                 additional parameters.
            weight (float): weight parameter to be used as weight1.

        Returns:
            np.ndarray: shape (..., 1) – expectation value of σ_y.
        """
        # Prepare parameter bindings
        bindings = [
            {self.input1: float(inp[0]), self.weight1: weight}
            for inp in inputs
        ]
        results = self.estimator_qnn.predict(bindings)
        return results.reshape(-1, 1)

# Expose for import
__all__ = ["HybridFCL"]
