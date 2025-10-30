"""Quantum convolutional neural network with adaptive pooling and entangled feature maps."""
import numpy as np
import torch

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA


def QCNNGen168(
    num_qubits: int = 8,
    depth: int = 3,
    entangle_feature_map: bool = True,
    entangle_pooling: bool = True,
) -> EstimatorQNN:
    """
    Construct a QCNN with adaptive pooling and optional entanglement.
    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int
        Number of convolution‑pooling pairs.
    entangle_feature_map : bool
        Whether the feature map uses full entanglement.
    entangle_pooling : bool
        Whether pooling layers use entanglement before measurement.
    Returns
    -------
    EstimatorQNN
        A QNN ready for hybrid training.
    """
    estimator = Estimator()

    # Feature map
    feature_map = ZFeatureMap(
        num_qubits,
        reps=2,
        entanglement="full" if entangle_feature_map else "linear",
    )

    # Ansatz
    ansatz = QuantumCircuit(num_qubits, name="Ansatz")
    for d in range(depth):
        ansatz.compose(conv_layer(num_qubits, f"c{d+1}"), inplace=True)
        ansatz.compose(
            pool_layer(num_qubits, f"p{d+1}", entangle_pooling),
            inplace=True,
        )

    # Combine feature map and ansatz
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn


def conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Two‑qubit convolution block with parameterised rotations."""
    qc = QuantumCircuit(num_qubits, name="Convolution")
    params = ParameterVector(prefix, length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        sub = QuantumCircuit(2)
        sub.rz(-np.pi / 2, 1)
        sub.cx(1, 0)
        sub.rz(params[i * 3], 0)
        sub.ry(params[i * 3 + 1], 1)
        sub.cx(0, 1)
        sub.ry(params[i * 3 + 2], 1)
        sub.cx(1, 0)
        sub.rz(np.pi / 2, 0)
        qc.append(sub.to_instruction(), [i, i + 1])
        qc.barrier()
    return qc


def pool_layer(
    num_qubits: int,
    prefix: str,
    entangle: bool = False,
) -> QuantumCircuit:
    """Adaptive pooling: measure one qubit and re‑initialize it."""
    qc = QuantumCircuit(num_qubits, name="Pooling")
    params = ParameterVector(prefix, length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        sub = QuantumCircuit(2)
        sub.rz(-np.pi / 2, 1)
        sub.cx(1, 0)
        sub.rz(params[i * 3], 0)
        sub.ry(params[i * 3 + 1], 1)
        sub.cx(0, 1)
        sub.ry(params[i * 3 + 2], 1)
        qc.append(sub.to_instruction(), [i, i + 1])
        qc.barrier()
        # Measure and reset the first qubit of the pair
        qc.measure(i, i)
        qc.reset(i)
    return qc


__all__ = ["QCNNGen168", "conv_layer", "pool_layer"]
