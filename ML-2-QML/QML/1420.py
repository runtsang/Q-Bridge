from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

class EstimatorQNNGen384:
    """
    A quantum neural network that mirrors the classical EstimatorQNNGen384.
    It uses a 3‑qubit feature map followed by a variational circuit with
    entanglement, and measures several Pauli observables to produce a
    regression output.  The class exposes the same interface as the
    classical version for seamless swapping in experiments.
    """

    def __init__(self, feature_map_depth: int = 2, var_depth: int = 3, num_qubits: int = 3) -> None:
        """
        Args:
            feature_map_depth: depth of the ZZFeatureMap.
            var_depth: depth of the variational RealAmplitudes circuit.
            num_qubits: number of qubits in the circuit.
        """
        # Build the feature map
        self.feature_map = ZZFeatureMap(
            feature_dimension=num_qubits,
            reps=feature_map_depth,
            entanglement='linear',
            paulis='zz',
            insert_barriers=True,
        )

        # Build the variational circuit
        self.var_circuit = RealAmplitudes(
            num_qubits=num_qubits,
            reps=var_depth,
            entanglement='full',
            insert_barriers=True,
        )

        # Combine into a single circuit
        self.circuit = QuantumCircuit(num_qubits)
        self.circuit.compose(self.feature_map, inplace=True)
        self.circuit.compose(self.var_circuit, inplace=True)

        # Observables: measure expectation of Y on each qubit
        self.observables = SparsePauliOp.from_list(
            [(f"Y" * num_qubits, 1.0)]
        )

        # Estimator primitive
        self.estimator = StatevectorEstimator()

        # Wrap into EstimatorQNN
        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=self.feature_map.parameters,
            weight_params=self.var_circuit.parameters,
            estimator=self.estimator,
        )

    def __call__(self, inputs: list[list[float]]) -> list[float]:
        """
        Evaluate the quantum neural network on a batch of inputs.
        Each input is a list of floats of length `num_qubits`.
        """
        return self.estimator_qnn.predict(inputs)

__all__ = ["EstimatorQNNGen384"]
