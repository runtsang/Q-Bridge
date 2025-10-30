import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.circuit.library import ZFeatureMap
from qiskit_machine_learning.utils import algorithm_globals
from QuantumClassifierModel import build_classifier_circuit

class QCNNHybrid:
    """Quantum‑enhanced QCNN that mirrors the classical architecture using a variational ansatz."""
    def __init__(self, seed: int = 12345, depth: int = 3) -> None:
        # Seed the backend
        algorithm_globals.random_seed = seed
        # Quantum estimator
        self.estimator = StatevectorEstimator()
        # Classical feature map embedded in the circuit
        self.feature_map = ZFeatureMap(8)
        # Build the variational classifier circuit
        circuit, encoding, weights, observables = build_classifier_circuit(8, depth)
        # Embed the classical feature map before the variational layers
        circuit.compose(self.feature_map, range(8), inplace=True)
        # Construct the EstimatorQNN
        self.qnn = EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observables,
            input_params=encoding,
            weight_params=weights,
            estimator=self.estimator,
        )

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the quantum circuit and return the sigmoid of the first observable,
        matching the binary classification interface of the classical counterpart.
        """
        preds = self.qnn.predict(inputs)
        return 1 / (1 + np.exp(-preds[:, 0]))

__all__ = ["QCNNHybrid"]
