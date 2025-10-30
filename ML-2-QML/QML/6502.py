"""QCNNPlus: Quantum QCNN with shuffled convolution schedule and shared parameters.

This module constructs a variational quantum circuit that mimics the
convolution‑pooling structure of the classical QCNN.  The key
enhancements are:

* **Parameter‑shuffled convolution** – the order of the Rz and Ry
  rotations inside each 2‑qubit convolution block is permuted
  (deterministically) to test the robustness of the circuit to
  gate‑ordering variations while keeping the logical connectivity
  identical.
* **Quantum‑reduced ansatz** – all convolution layers re‑use the
  same ParameterVector, dramatically cutting the total number of
  learnable parameters and the circuit depth.

The resulting :class:`EstimatorQNN` can be used directly with
``qiskit_machine_learning`` classifiers.

Typical usage::

    qnn = QCNNPlus()
    # qnn is an EstimatorQNN instance ready for training

"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from sklearn.model_selection import train_test_split

# deterministic seed for reproducibility
algorithm_globals.random_seed = 12345

def _shuffled_conv_circuit(params: ParameterVector, shuffle_order: tuple[int, int, int]) -> QuantumCircuit:
    """Return a 2‑qubit convolution circuit with a deterministic shuffle of rotations.

    Parameters
    ----------
    params
        A ParameterVector of length 3.
    shuffle_order
        A tuple of indices (0‑based) indicating the order in which the
        Rz, Ry, and Rz gates are applied.

    Returns
    -------
    QuantumCircuit
        The 2‑qubit convolution block.

    """
    target = QuantumCircuit(2)
    # base sequence: Rz(-π/2), CX, Rz, Ry, CX, Ry, CX, Rz(π/2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    # rotation gates applied in the order specified by shuffle_order
    # mapping: 0 -> Rz, 1 -> Ry, 2 -> Rz (the second Rz)
    if shuffle_order[0] == 0:
        target.rz(params[0], 0)
    elif shuffle_order[0] == 1:
        target.ry(params[0], 0)
    else:
        target.rz(params[0], 0)
    if shuffle_order[1] == 0:
        target.rz(params[1], 1)
    elif shuffle_order[1] == 1:
        target.ry(params[1], 1)
    else:
        target.rz(params[1], 1)
    target.cx(0, 1)
    if shuffle_order[2] == 0:
        target.rz(params[2], 1)
    elif shuffle_order[2] == 1:
        target.ry(params[2], 1)
    else:
        target.rz(params[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    return target

def conv_layer(num_qubits: int, param_vector: ParameterVector, shuffle_order: tuple[int, int, int]) -> QuantumCircuit:
    """Build a convolutional layer where each 2‑qubit block shares the same parameter vector.

    Parameters
    ----------
    num_qubits
        Total number of qubits in the layer.
    param_vector
        ParameterVector of length 3 used for every 2‑qubit block.
    shuffle_order
        Order in which rotation gates are applied inside each block.

    Returns
    -------
    QuantumCircuit
        The convolutional layer as a QuantumCircuit.

    """
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    # pairwise convolution on odd‑even qubit pairs
    for i in range(0, num_qubits - 1, 2):
        block = _shuffled_conv_circuit(param_vector, shuffle_order)
        qc.append(block, [qubits[i], qubits[i + 1]])
        qc.barrier()
    # second pass: shift by one to cover overlapping pairs
    for i in range(1, num_qubits - 1, 2):
        block = _shuffled_conv_circuit(param_vector, shuffle_order)
        qc.append(block, [qubits[i], qubits[i + 1]])
        qc.barrier()
    return qc

def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """2‑qubit pooling block used in both pooling layers."""
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    return target

def pool_layer(sources: list[int], sinks: list[int], param_vector: ParameterVector) -> QuantumCircuit:
    """Pooling layer that maps ``sources`` qubits into ``sinks`` qubits.

    Parameters
    ----------
    sources
        List of source qubit indices.
    sinks
        List of sink qubit indices.
    param_vector
        ParameterVector of length 3 used for every 2‑qubit block.

    Returns
    -------
    QuantumCircuit
        The pooling layer as a QuantumCircuit.

    """
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    for src, sink in zip(sources, sinks):
        block = pool_circuit(param_vector)
        qc.append(block, [src, sink])
        qc.barrier()
    return qc

def QCNNPlus() -> EstimatorQNN:
    """Build and return an EstimatorQNN that implements the QCNNPlus ansatz.

    Returns
    -------
    EstimatorQNN
        A variational quantum neural network ready for training.

    """
    estimator = Estimator()
    # shared parameter vector for all convolution layers
    conv_params = ParameterVector("c", length=3)
    # shared parameter vector for all pooling layers
    pool_params = ParameterVector("p", length=3)
    # deterministic shuffle order for the convolution rotations
    shuffle_order = (1, 0, 2)  # Ry, Rz, Rz

    # Feature map (8‑qubit Z feature map)
    feature_map = ZFeatureMap(8)

    # Ansatz construction
    ansatz = QuantumCircuit(8, name="Ansatz")

    # First Convolution + Pooling
    ansatz.compose(conv_layer(8, conv_params, shuffle_order), list(range(8)), inplace=True)
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], pool_params), list(range(8)), inplace=True)

    # Second Convolution + Pooling (reusing the same parameters)
    ansatz.compose(conv_layer(4, conv_params, shuffle_order), list(range(4, 8)), inplace=True)
    ansatz.compose(pool_layer([0, 1], [2, 3], pool_params), list(range(4, 8)), inplace=True)

    # Third Convolution + Pooling
    ansatz.compose(conv_layer(2, conv_params, shuffle_order), list(range(6, 8)), inplace=True)
    ansatz.compose(pool_layer([0], [1], pool_params), list(range(6, 8)), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = ["QCNNPlus"]
