"""
QCNNAutoencoder: Quantum implementation mirroring the classical hybrid.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals


def QCNNAutoencoder() -> EstimatorQNN:
    """
    Constructs a quantum neural network that:
      1. Encodes data with a ZFeatureMap (serving as a quantum autoencoder).
      2. Applies a RealAmplitudes ansatz (variational layer) on the same qubits.
      3. Executes a QCNN‑style convolution and pooling sequence.
    Returns an EstimatorQNN ready for variational training.
    """

    algorithm_globals.random_seed = 12345
    estimator = Estimator()

    # ----------------------------------------------------------- #
    #  Convolutional block (2‑qubit entangling gate pattern)
    # ----------------------------------------------------------- #
    def conv_circuit(params: ParameterVector) -> QuantumCircuit:
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

    # ----------------------------------------------------------- #
    #  Pooling block (2‑qubit partial entanglement)
    # ----------------------------------------------------------- #
    def pool_circuit(params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    # ----------------------------------------------------------- #
    #  Helper to build a convolutional layer over many qubits
    # ----------------------------------------------------------- #
    def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for i in range(0, num_qubits, 2):
            sub = conv_circuit(params[i // 2 * 3 : i // 2 * 3 + 3])
            qc.append(sub.to_instruction(), [i, i + 1])
        return qc

    # ----------------------------------------------------------- #
    #  Helper to build a pooling layer over many qubits
    # ----------------------------------------------------------- #
    def pool_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for i in range(0, num_qubits, 2):
            sub = pool_circuit(params[i // 2 * 3 : i // 2 * 3 + 3])
            qc.append(sub.to_instruction(), [i, i + 1])
        return qc

    # ----------------------------------------------------------- #
    #  Feature map (acts as a quantum autoencoder)
    # ----------------------------------------------------------- #
    feature_map = ZFeatureMap(8)

    # ----------------------------------------------------------- #
    #  Variational ansatz (RealAmplitudes) – serves as the quantum encoder
    # ----------------------------------------------------------- #
    ansatz = RealAmplitudes(8, reps=3)

    # ----------------------------------------------------------- #
    #  QCNN layers: 3 conv + 3 pool
    # ----------------------------------------------------------- #
    conv1 = conv_layer(8, "c1")
    pool1 = pool_layer(8, "p1")

    conv2 = conv_layer(4, "c2")
    pool2 = pool_layer(4, "p2")

    conv3 = conv_layer(2, "c3")
    pool3 = pool_layer(2, "p3")

    # ----------------------------------------------------------- #
    #  Assemble the full circuit
    # ----------------------------------------------------------- #
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)
    circuit.compose(conv1, range(8), inplace=True)
    circuit.compose(pool1, range(8), inplace=True)
    circuit.compose(conv2, range(4, 8), inplace=True)
    circuit.compose(pool2, range(4, 8), inplace=True)
    circuit.compose(conv3, range(6, 8), inplace=True)
    circuit.compose(pool3, range(6, 8), inplace=True)

    # Observable: single‑qubit Z on qubit 0
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # ----------------------------------------------------------- #
    #  Wrap into an EstimatorQNN
    # ----------------------------------------------------------- #
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters
        + conv1.parameters
        + pool1.parameters
        + conv2.parameters
        + pool2.parameters
        + conv3.parameters
        + pool3.parameters,
        estimator=estimator,
    )
    return qnn


__all__ = ["QCNNAutoencoder"]
