"""QCNNHybrid: quantum implementation of the QCNN architecture.

The class builds a parameterised ansatz that mirrors the convolutional
and pooling layers of the classical QCNN.  It uses a ZFeatureMap for
data encoding and a variational ansatz composed of repeated
convolution–pooling blocks.  The resulting circuit is wrapped in an
EstimatorQNN so that it can be trained with standard gradient‑based
optimisers.  The network outputs a two‑class probability vector
obtained from a Z‑observable measurement on the first qubit.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

class QCNNHybrid:
    """Quantum QCNN implementation."""

    def __init__(self, num_qubits: int = 8, depth: int = 3,
                 backend=None, shots: int = 1024) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = backend
        self.shots = shots
        algorithm_globals.random_seed = 12345
        self.estimator = Estimator()
        self.circuit, self.input_params, self.weight_params, self.observables = self._build_circuit()
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observables,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def _build_circuit(self) -> tuple[QuantumCircuit, list, list, list[SparsePauliOp]]:
        """Construct the QCNN ansatz with feature map, convolution and pooling."""
        # Feature map
        feature_map = self._build_feature_map()

        # Ansatz
        ansatz = QuantumCircuit(self.num_qubits, name="Ansatz")
        ansatz.compose(self._conv_layer(self.num_qubits, "c1"), range(self.num_qubits), inplace=True)
        ansatz.compose(self._pool_layer(list(range(self.num_qubits // 2)),
                                        list(range(self.num_qubits // 2, self.num_qubits)),
                                        "p1"), range(self.num_qubits), inplace=True)
        ansatz.compose(self._conv_layer(self.num_qubits // 2, "c2"),
                       range(self.num_qubits // 2, self.num_qubits), inplace=True)
        ansatz.compose(self._pool_layer(list(range(self.num_qubits // 4)),
                                        list(range(self.num_qubits // 4, self.num_qubits // 2)),
                                        "p2"), range(self.num_qubits // 2, self.num_qubits), inplace=True)
        ansatz.compose(self._conv_layer(self.num_qubits // 4, "c3"),
                       range(self.num_qubits // 4 * 3, self.num_qubits), inplace=True)
        ansatz.compose(self._pool_layer([self.num_qubits // 4 * 3],
                                        [self.num_qubits // 4 * 3 + 1],
                                        "p3"), range(self.num_qubits // 4 * 3, self.num_qubits), inplace=True)

        # Combine feature map and ansatz
        circuit = QuantumCircuit(self.num_qubits)
        circuit.compose(feature_map, range(self.num_qubits), inplace=True)
        circuit.compose(ansatz, range(self.num_qubits), inplace=True)

        # Observables: Z on first qubit to produce scalar expectation
        observable = SparsePauliOp.from_list([("Z" + "I" * (self.num_qubits - 1), 1)])

        # Parameter lists
        input_params = list(feature_map.parameters)
        weight_params = list(ansatz.parameters)
        return circuit, input_params, weight_params, [observable]

    def _build_feature_map(self) -> QuantumCircuit:
        """ZFeatureMap implemented with ParameterVector."""
        params = ParameterVector("x", self.num_qubits)
        feature_map = QuantumCircuit(self.num_qubits, name="ZFeatureMap")
        for q, p in enumerate(params):
            feature_map.rz(p, q)
            feature_map.ry(p, q)
            feature_map.rz(p, q)
        return feature_map

    def _conv_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        """Convolutional block with 3‑parameter two‑qubit unitary."""
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        param_index = 0
        for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
            sub = self._conv_circuit(params[param_index:param_index + 3], q1, q2)
            qc.append(sub, [q1, q2])
            qc.barrier()
            param_index += 3
        return qc

    def _conv_circuit(self, params, q1, q2) -> QuantumCircuit:
        """Two‑qubit unitary used in convolution."""
        sub = QuantumCircuit(2, name="ConvUnit")
        sub.rz(-np.pi / 2, q2)
        sub.cx(q2, q1)
        sub.rz(params[0], q1)
        sub.ry(params[1], q2)
        sub.cx(q1, q2)
        sub.ry(params[2], q2)
        sub.cx(q2, q1)
        sub.rz(np.pi / 2, q1)
        return sub

    def _pool_layer(self, sources, sinks, param_prefix: str) -> QuantumCircuit:
        """Pooling block that maps two qubits to one."""
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        params = ParameterVector(param_prefix, length=len(sources) * 3)
        for src, sink in zip(sources, sinks):
            sub = self._pool_circuit(params[:3], src, sink)
            qc.append(sub, [src, sink])
            qc.barrier()
            params = params[3:]
        return qc

    def _pool_circuit(self, params, src, sink) -> QuantumCircuit:
        """Two‑qubit unitary used in pooling."""
        sub = QuantumCircuit(2, name="PoolUnit")
        sub.rz(-np.pi / 2, sink)
        sub.cx(sink, src)
        sub.rz(params[0], src)
        sub.ry(params[1], sink)
        sub.cx(src, sink)
        sub.ry(params[2], sink)
        return sub

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Run the QCNN and return a probability vector."""
        exp = self.qnn(inputs).reshape(-1)
        probs = 1 / (1 + np.exp(-exp))
        return np.stack([probs, 1 - probs], axis=-1)

    def parameters(self):
        """Return trainable parameter names."""
        return self.weight_params
