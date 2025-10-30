import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StateEstimator
from qiskit.quantum_info import SparsePauliOp
from typing import Iterable

class HybridFullyConnectedLayer:
    """
    Quantum implementation of a fully‑connected layer inspired by
    EstimatorQNN.  The circuit contains a single qubit with H, Ry,
    and Rx gates where the Ry gate encodes the input and the Rx gate
    represents the weight.  The expectation value of the Y Pauli
    operator is returned, matching the output shape of the classical
    counterpart.
    """

    def __init__(self, backend=None, shots: int = 1024) -> None:
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

        # Define parameters
        self.input_param = Parameter("θ_in")
        self.weight_param = Parameter("θ_w")

        # Build circuit
        self.circuit = QuantumCircuit(1)
        self.circuit.h(0)
        self.circuit.ry(self.input_param, 0)
        self.circuit.rx(self.weight_param, 0)
        self.circuit.measure_all()

        # Observable for expectation value
        self.observable = SparsePauliOp.from_list([("Y", 1)])

        # EstimatorQNN wrapper
        self.estimator = StateEstimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=[self.input_param],
            weight_params=[self.weight_param],
            estimator=self.estimator,
        )

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit with the supplied parameters.
        `thetas` should contain two values: [input, weight].
        """
        param_binds = [{self.input_param: thetas[0], self.weight_param: thetas[1]}]
        result = self.qnn.predict(param_binds, n_shots=self.shots)
        return np.array(result)

__all__ = ["HybridFullyConnectedLayer"]
