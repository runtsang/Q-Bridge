"""Quantum implementation of the QCNN hybrid architecture.

The circuit combines a ZFeatureMap, a convolution‑pooling ansatz mimicking
the classical layers, and a final classifier ansatz that applies a
variational rotation per qubit followed by CZ couplings, mirroring the
classical classifier construction in `build_classifier_circuit`.  The
observable set contains a Z operator on each qubit, enabling a
probability‑based prediction for binary classification.

The wrapper class QCNNHybrid provides a convenient interface to
instantiate the EstimatorQNN and expose a `predict` method.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit import algorithm_globals

class QCNNHybrid:
    """Quantum analog of the classical QCNNHybrid model."""
    def __init__(self, seed: int = 12345) -> None:
        self.seed = seed
        self.estimator = Estimator()
        self.qnn = self._build_qnn()

    def _build_qnn(self) -> EstimatorQNN:
        """Build the full ansatz combining convolution, pooling and classifier layers."""
        np.random.seed(self.seed)
        algorithm_globals.random_seed = self.seed

        # ---- Feature map ----
        feature_map = ZFeatureMap(8)

        # ---- Convolution and pooling primitives ----
        def conv_circuit(params):
            qc = QuantumCircuit(2)
            qc.rz(-np.pi / 2, 1)
            qc.cx(1, 0)
            qc.rz(params[0], 0)
            qc.ry(params[1], 1)
            qc.cx(0, 1)
            qc.ry(params[2], 1)
            qc.cx(1, 0)
            qc.rz(np.pi / 2, 0)
            return qc

        def pool_circuit(params):
            qc = QuantumCircuit(2)
            qc.rz(-np.pi / 2, 1)
            qc.cx(1, 0)
            qc.rz(params[0], 0)
            qc.ry(params[1], 1)
            qc.cx(0, 1)
            qc.ry(params[2], 1)
            return qc

        def conv_layer(num_qubits, prefix):
            qc = QuantumCircuit(num_qubits)
            params = ParameterVector(prefix, length=num_qubits // 2 * 3)
            idx = 0
            for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
                qc.compose(conv_circuit(params[idx:idx+3]), [q1, q2], inplace=True)
                idx += 3
            return qc

        def pool_layer(num_qubits, prefix):
            qc = QuantumCircuit(num_qubits)
            params = ParameterVector(prefix, length=num_qubits // 2 * 3)
            idx = 0
            for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
                qc.compose(pool_circuit(params[idx:idx+3]), [q1, q2], inplace=True)
                idx += 3
            return qc

        # ---- Build ansatz ----
        ansatz = QuantumCircuit(8)
        ansatz.compose(conv_layer(8, "c1"), inplace=True)
        ansatz.compose(pool_layer(8, "p1"), inplace=True)
        ansatz.compose(conv_layer(4, "c2"), inplace=True)
        ansatz.compose(pool_layer(4, "p2"), inplace=True)
        ansatz.compose(conv_layer(2, "c3"), inplace=True)
        ansatz.compose(pool_layer(2, "p3"), inplace=True)

        # ---- Classifier ansatz (mirrors build_classifier_circuit) ----
        num_qubits = 8
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * 5)  # depth = 5

        classifier = QuantumCircuit(num_qubits)
        for qubit, param in zip(range(num_qubits), encoding):
            classifier.rx(param, qubit)

        idx = 0
        for _ in range(5):
            for qubit in range(num_qubits):
                classifier.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                classifier.cz(qubit, qubit + 1)

        # ---- Combine all ----
        circuit = QuantumCircuit(8)
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)
        circuit.compose(classifier, inplace=True)

        # ---- Observables ----
        observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]

        return EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observables,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters + classifier.parameters,
            estimator=self.estimator,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the expectation values of the observables for the given input."""
        return self.qnn.predict(inputs=X)

    def weight_sizes(self) -> list[int]:
        """Return a list of the number of parameters per layer in the quantum circuit."""
        return [p.numel() for p in self.qnn.circuit.parameters]

__all__ = ["QCNNHybrid"]
