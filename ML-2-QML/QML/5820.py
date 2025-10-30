"""QCNNGen216 – Quantum implementation inspired by QCNN and QuantumClassifierModel."""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution unit used in the QCNN."""
    qc = QuantumCircuit(2)
    qc.rz(-3.141592653589793 / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(3.141592653589793 / 2, 0)
    return qc


def _conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Convolutional layer that applies _conv_circuit to all adjacent qubit pairs."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for i in range(0, num_qubits - 1, 2):
        sub = _conv_circuit(params[i // 2 * 3 : i // 2 * 3 + 3])
        qc.append(sub.to_instruction(), [i, i + 1])
        qc.barrier()
    return qc


def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling unit used in the QCNN."""
    qc = QuantumCircuit(2)
    qc.rz(-3.141592653589793 / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def _pool_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Pooling layer that applies _pool_circuit to all adjacent qubit pairs."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for i in range(0, num_qubits - 1, 2):
        sub = _pool_circuit(params[i // 2 * 3 : i // 2 * 3 + 3])
        qc.append(sub.to_instruction(), [i, i + 1])
        qc.barrier()
    return qc


class QCNNGen216:
    """
    Quantum circuit that mirrors the classical QCNNGen216 architecture.
    It composes a feature map, a stack of convolutional and pooling layers,
    and a variational ansatz that includes depth‑controlled encoding and
    entangling gates.  The resulting EstimatorQNN can be trained by classical
    optimizers.
    """

    def __init__(self, num_qubits: int = 8, depth: int = 3) -> None:
        self.num_qubits = num_qubits
        self.depth = depth

        # Feature map
        self.feature_map = ZFeatureMap(num_qubits)

        # Variational ansatz: depth‑controlled encoding + entangling layers
        self.ansatz = self._build_ansatz(num_qubits, depth)

        # Full circuit
        self.circuit = QuantumCircuit(num_qubits)
        self.circuit.compose(self.feature_map, range(num_qubits), inplace=True)
        self.circuit.compose(self.ansatz, range(num_qubits), inplace=True)

        # Observable: single‑qubit Z on the first qubit
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

        # Estimator and QNN
        self.estimator = StatevectorEstimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

        # Metadata for side‑by‑side comparison
        self.encoding = ParameterVector("x", num_qubits)
        self.weight_params = self.ansatz.parameters
        self.observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]

    def _build_ansatz(self, num_qubits: int, depth: int) -> QuantumCircuit:
        """Builds a layered ansatz that alternates convolutional, pooling, and
        simple encoding/entanglement blocks."""
        ansatz = QuantumCircuit(num_qubits)

        # Encoding layer
        for q in range(num_qubits):
            ansatz.rx(ParameterVector("x", num_qubits)[q], q)

        # Depth‑controlled layers
        for d in range(depth):
            ansatz.compose(_conv_layer(num_qubits, f"c{d}"), range(num_qubits), inplace=True)
            ansatz.compose(_pool_layer(num_qubits, f"p{d}"), range(num_qubits), inplace=True)

        return ansatz

    def get_circuit(self) -> QuantumCircuit:
        """Return the full, decomposed circuit."""
        return self.circuit.decompose()

    def get_qnn(self) -> EstimatorQNN:
        """Return the EstimatorQNN ready for training."""
        return self.qnn

    def get_metadata(self) -> Tuple[List[int], List[int], List[int]]:
        """Return encoding indices, weight parameter sizes, and observables."""
        weight_sizes = [len(self.weight_params)]
        return list(self.encoding), weight_sizes, [len(self.observables)]

__all__ = ["QCNNGen216"]
