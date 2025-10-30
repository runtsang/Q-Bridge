"""Quantum hybrid model that mirrors the classical structure but replaces
the CNN encoder and QCNN block with quantum circuits.

The class builds a Z‑feature‑map, a QCNN‑style ansatz composed of
convolution and pooling layers, and a variational classifier layer.
It returns the expectation values of Pauli‑Z observables, which
serve as a classical feature vector for downstream tasks.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator


def conv_circuit(params: ParameterVector, qubits: list[int]) -> QuantumCircuit:
    """Two‑qubit convolution unit used in QCNN."""
    qc = QuantumCircuit(len(qubits))
    # Unpack parameters for the two qubits
    q1, q2 = qubits
    qc.rz(-np.pi / 2, q2)
    qc.cx(q2, q1)
    qc.rz(params[0], q1)
    qc.ry(params[1], q2)
    qc.cx(q1, q2)
    qc.ry(params[2], q2)
    qc.cx(q2, q1)
    qc.rz(np.pi / 2, q1)
    return qc


def pool_circuit(params: ParameterVector, qubits: list[int]) -> QuantumCircuit:
    """Two‑qubit pooling unit used in QCNN."""
    qc = QuantumCircuit(len(qubits))
    q1, q2 = qubits
    qc.rz(-np.pi / 2, q2)
    qc.cx(q2, q1)
    qc.rz(params[0], q1)
    qc.ry(params[1], q2)
    qc.cx(q1, q2)
    qc.ry(params[2], q2)
    return qc


def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Convolutional layer composed of multiple two‑qubit conv units."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        sub = conv_circuit(params[idx : idx + 3], [q1, q2])
        qc.append(sub, [q1, q2])
        idx += 3
    return qc


def pool_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Pooling layer composed of multiple two‑qubit pool units."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        sub = pool_circuit(params[idx : idx + 3], [q1, q2])
        qc.append(sub, [q1, q2])
        idx += 3
    return qc


def build_qcnn_ansatz(num_qubits: int, depth: int) -> QuantumCircuit:
    """
    Build a QCNN‑style ansatz with `depth` convolution + pooling stages.
    The ansatz is fully parameterised; each layer receives its own parameters.
    """
    qc = QuantumCircuit(num_qubits)
    for d in range(depth):
        # Convolution
        qc.append(conv_layer(num_qubits, f"c{d}"), range(num_qubits))
        # Pooling
        qc.append(pool_layer(num_qubits, f"p{d}"), range(num_qubits))
    return qc


def build_classifier_circuit(num_qubits: int, depth: int) -> QuantumCircuit:
    """
    Build a variational classifier circuit similar to the classical
    feed‑forward classifier in the seed.  It encodes the input with RX,
    applies `depth` layers of Ry and CZ gates, and measures each qubit.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)
    qc = QuantumCircuit(num_qubits)

    # Encoding
    for i, q in enumerate(range(num_qubits)):
        qc.rx(encoding[i], q)

    # Variational layers
    w_idx = 0
    for _ in range(depth):
        for q in range(num_qubits):
            qc.ry(weights[w_idx], q)
            w_idx += 1
        for q in range(num_qubits - 1):
            qc.cz(q, q + 1)

    return qc


class QuantumNATHybrid:
    """
    Quantum implementation of the hybrid architecture.

    Parameters
    ----------
    num_qubits : int
        Number of qubits used to encode the 8‑dimensional input.
    conv_depth : int
        Number of convolution + pooling stages in the QCNN ansatz.
    classifier_depth : int
        Depth of the variational classifier.
    """

    def __init__(
        self,
        num_qubits: int = 8,
        conv_depth: int = 3,
        classifier_depth: int = 3,
    ) -> None:
        # Feature map for classical data
        self.feature_map = ZFeatureMap(num_qubits)

        # QCNN ansatz
        self.qcnn_ansatz = build_qcnn_ansatz(num_qubits, conv_depth)

        # Classifier ansatz
        self.classifier_ansatz = build_classifier_circuit(num_qubits, classifier_depth)

        # Combine into a single circuit
        self.circuit = QuantumCircuit(num_qubits)
        self.circuit.append(self.feature_map, range(num_qubits))
        self.circuit.append(self.qcnn_ansatz, range(num_qubits))
        self.circuit.append(self.classifier_ansatz, range(num_qubits))

        # Observables: Pauli‑Z on each qubit
        self.observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]

        # Estimator QNN
        estimator = Estimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=self.feature_map.parameters,
            weight_params=self.qcnn_ansatz.parameters + self.classifier_ansatz.parameters,
            estimator=estimator,
        )

    def forward(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Execute the quantum circuit on input data.

        Parameters
        ----------
        x : np.ndarray or torch.Tensor
            Input data of shape (batch, 8).  Each row is encoded into
            the qubits via the Z‑feature map.

        Returns
        -------
        np.ndarray
            Expectation values of the Z observables, shape (batch, num_qubits).
            These can be interpreted as a classical feature vector.
        """
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x, dtype=np.float64)

        return self.qnn(x_np)

    def circuit(self) -> QuantumCircuit:
        """Return the underlying quantum circuit for inspection."""
        return self.circuit


__all__ = ["QuantumNATHybrid"]
