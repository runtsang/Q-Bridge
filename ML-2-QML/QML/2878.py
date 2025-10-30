import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit import QuantumCircuit, transpile, assemble, Aer
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as EstimatorPrimitive

class HybridBinaryClassifier(nn.Module):
    """
    Quantum‑classical hybrid classifier that implements a QCNN‑style circuit.
    The network consists of:
        * A Z‑feature map that encodes the 8‑dimensional classical input.
        * Three convolutional layers (two‑qubit unitaries) followed by pooling layers.
        * An observable measuring Z on the first qubit.
        * The EstimatorQNN provides exact gradients via the parameter‑shift rule.
    The forward pass returns a probability distribution over two classes.
    """
    def __init__(self, shots: int = 1024) -> None:
        super().__init__()
        self.shots = shots
        self.backend = Aer.get_backend("aer_simulator")
        self.estimator = EstimatorPrimitive()

        # Build the QCNN circuit
        self.circuit = self._build_qcnn()
        self.observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
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

    def _conv_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc.append(self._conv_circuit(params[param_index:param_index + 3]), [q1, q2])
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc.append(self._conv_circuit(params[param_index:param_index + 3]), [q1, q2])
            qc.barrier()
            param_index += 3
        qc_inst = qc.to_instruction()
        return QuantumCircuit(num_qubits).append(qc_inst, qubits)

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _pool_layer(self, sources, sinks, param_prefix: str) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for source, sink in zip(sources, sinks):
            qc.append(self._pool_circuit(params[param_index:param_index + 3]), [source, sink])
            qc.barrier()
            param_index += 3
        qc_inst = qc.to_instruction()
        return QuantumCircuit(num_qubits).append(qc_inst, range(num_qubits))

    def _build_qcnn(self) -> QuantumCircuit:
        # Feature map
        self.feature_map = ZFeatureMap(8)
        # Ansatz construction
        self.ansatz = QuantumCircuit(8, name="Ansatz")

        # Layer 1
        self.ansatz.compose(self._conv_layer(8, "c1"), list(range(8)), inplace=True)
        self.ansatz.compose(self._pool_layer([0,1,2,3], [4,5,6,7], "p1"), list(range(8)), inplace=True)

        # Layer 2
        self.ansatz.compose(self._conv_layer(4, "c2"), list(range(4,8)), inplace=True)
        self.ansatz.compose(self._pool_layer([0,1], [2,3], "p2"), list(range(4,8)), inplace=True)

        # Layer 3
        self.ansatz.compose(self._conv_layer(2, "c3"), list(range(6,8)), inplace=True)
        self.ansatz.compose(self._pool_layer([0], [1], "p3"), list(range(6,8)), inplace=True)

        # Combine feature map and ansatz
        circuit = QuantumCircuit(8)
        circuit.compose(self.feature_map, range(8), inplace=True)
        circuit.compose(self.ansatz, range(8), inplace=True)
        return circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        :param inputs: Tensor of shape (batch, 8) containing raw features.
        :return: Tensor of shape (batch, 2) with class probabilities.
        """
        # Build input dictionary for EstimatorQNN
        input_dict = {param: inputs[:, idx] for idx, param in enumerate(self.qnn.input_params)}
        # EstimatorQNN returns expectation values in [-1, 1]
        expectation = self.qnn(input_dict).reshape(-1, 1)
        # Map to [0, 1] probability
        prob = (expectation + 1) / 2
        return torch.cat((prob, 1 - prob), dim=-1)

class QCNet(HybridBinaryClassifier):
    """Alias kept for backward compatibility with the original anchor."""
    pass

__all__ = ["HybridBinaryClassifier", "QCNet"]
