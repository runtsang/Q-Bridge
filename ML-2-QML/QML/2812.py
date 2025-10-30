"""Quantum implementation of the QCNN ansatz with a variational classifier.

The module defines:
* `QCNNHybrid` – a class that encapsulates a qiskit circuit, a StatevectorEstimator,
  and an EstimatorQNN.  It exposes a `forward` method that evaluates the circuit
  on classical feature vectors.
* `build_classifier_circuit` – a helper that constructs a simple layered ansatz
  with explicit encoding and variational parameters, mirroring the classical
  helper function.
"""

from __future__ import annotations

from typing import Tuple, Iterable, List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit.circuit.library import ZFeatureMap
from qiskit_machine_learning.neural_networks import EstimatorQNN


class QCNNHybrid:
    """Quantum QCNN with a variational ansatz and a single Z observable.

    The circuit follows the same convolution–pooling pattern as the classical model:
    a Z‑feature map, a stack of depth‑controlled Ry rotations (convolution) and
    CZ gates (pooling).  The class exposes a `forward` method that returns the
    expectation value of the observable for a batch of input feature vectors.
    """
    def __init__(self, num_qubits: int = 8, depth: int = 3, seed: int = 12345) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.estimator = Estimator()
        self.circuit = self._build_qnn()
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    def _build_qnn(self) -> QuantumCircuit:
        # Feature map
        self.feature_map = ZFeatureMap(self.num_qubits, reps=1, entanglement="full")

        # Variational ansatz
        self.ansatz = QuantumCircuit(self.num_qubits)
        weight_params = ParameterVector("theta", self.num_qubits * self.depth)

        for d in range(self.depth):
            # Convolution layer: Ry rotations
            for q in range(self.num_qubits):
                self.ansatz.ry(weight_params[d * self.num_qubits + q], q)
            # Pooling layer: CZ gates between adjacent qubits
            if d < self.depth - 1:
                for q in range(0, self.num_qubits - 1, 2):
                    self.ansatz.cz(q, q + 1)

        # Observable: single Z on the first qubit
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (self.num_qubits - 1), 1)])

        # Combine feature map and ansatz
        circuit = QuantumCircuit(self.num_qubits)
        circuit.append(self.feature_map, range(self.num_qubits))
        circuit.append(self.ansatz, range(self.num_qubits))
        return circuit.decompose()

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Evaluate the QNN on a batch of feature vectors."""
        return np.array(self.qnn.predict(inputs))

    @property
    def parameters(self) -> Iterable[np.ndarray]:
        """Return the trainable parameters of the ansatz."""
        return self.ansatz.parameters


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Construct a simple layered ansatz with explicit encoding and variational parameters.

    The returned tuple consists of:
    * the circuit (QuantumCircuit)
    * the encoding parameters (ParameterVector)
    * the weight parameters (ParameterVector)
    * a list of observables (SparsePauliOp)
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    index = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables


__all__ = ["QCNNHybrid", "build_classifier_circuit"]
