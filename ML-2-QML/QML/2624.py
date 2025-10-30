"""Quantum implementation of QCNNGen182.

This module defines a QCNNGen182 class that builds a quantum circuit
combining a quantum convolutional filter (from Conv.py QML seed) with
the QCNN ansatz (from QCNN.py QML seed).  The feature map encodes the
input data into a unitary that first applies a 2×2 quantum filter
followed by a ZFeatureMap on all qubits.  The ansatz consists of
convolutional and pooling layers that operate on the full register.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.circuit.random import random_circuit

# --------------------------------------------------------------------------- #
#  Quantum convolutional filter (inspired by Conv.py QML seed)
# --------------------------------------------------------------------------- #
def _conv_filter_circuit(kernel_size: int = 2, threshold: float = 0.5) -> QuantumCircuit:
    """Return a sub‑circuit that implements a 2×2 quantum filter.

    Parameters
    ----------
    kernel_size : int
        Size of the kernel (default 2 → 4 qubits).
    threshold : float
        Threshold used to decide whether a data value triggers a π rotation.

    Returns
    -------
    QuantumCircuit
        Sub‑circuit with ``kernel_size**2`` qubits and ``kernel_size**2``
        parameterized rotations.
    """
    n_qubits = kernel_size * kernel_size
    qc = QuantumCircuit(n_qubits)
    theta = ParameterVector("θ", length=n_qubits)
    for i in range(n_qubits):
        qc.rx(theta[i], i)
    qc.barrier()
    qc += random_circuit(n_qubits, depth=2, seed=42)
    return qc

# --------------------------------------------------------------------------- #
#  2‑qubit convolution circuit (from QCNN.py QML seed)
# --------------------------------------------------------------------------- #
def _conv_circuit_2q(params: ParameterVector) -> QuantumCircuit:
    """2‑qubit convolution unitary with 3 parameters."""
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    return target

# --------------------------------------------------------------------------- #
#  Convolution and pooling layers (from QCNN.py QML seed)
# --------------------------------------------------------------------------- #
def _conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Convolutional layer for the QCNN ansatz."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    # Pairwise convolution
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        sub = _conv_circuit_2q(params[param_index : param_index + 3])
        qc.append(sub, [q1, q2])
        qc.barrier()
        param_index += 3
    # Wrap‑around convolution
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        sub = _conv_circuit_2q(params[param_index : param_index + 3])
        qc.append(sub, [q1, q2])
        qc.barrier()
        param_index += 3
    return qc

def _pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    """Pooling layer that discards source qubits into sinks."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=(num_qubits // 2) * 3)
    for source, sink in zip(sources, sinks):
        sub = _conv_circuit_2q(params[param_index : param_index + 3])
        qc.append(sub, [source, sink])
        qc.barrier()
        param_index += 3
    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc

# --------------------------------------------------------------------------- #
#  Feature map combining quantum filter and ZFeatureMap
# --------------------------------------------------------------------------- #
def _custom_feature_map(num_qubits: int = 8, threshold: float = 0.5) -> QuantumCircuit:
    """Custom feature map that first applies a 2×2 quantum filter and then a ZFeatureMap.

    The map uses ``num_qubits`` parameters: the first ``kernel_size**2`` are used
    for the filter, the remaining for the ZFeatureMap.
    """
    kernel_size = 2
    filter_qubits = kernel_size * kernel_size
    if num_qubits < filter_qubits:
        raise ValueError("num_qubits must be at least 4 for the filter")
    qc = QuantumCircuit(num_qubits)
    # Parameters
    params = ParameterVector("x", length=num_qubits)
    # 1. Quantum filter on first 4 qubits
    filter_circ = _conv_filter_circuit(kernel_size=kernel_size, threshold=threshold)
    filter_circ = filter_circ.copy()
    filter_circ.assign_parameters(params[:filter_qubits], qubits=list(range(filter_qubits)))
    qc.append(filter_circ, range(filter_qubits))
    # 2. ZFeatureMap on all qubits
    zfm = ZFeatureMap(num_qubits, reps=1, entanglement="full")
    zfm = zfm.copy()
    zfm.assign_parameters(params, qubits=list(range(num_qubits)))
    qc.append(zfm, range(num_qubits))
    return qc

# --------------------------------------------------------------------------- #
#  QCNNGen182 quantum circuit builder
# --------------------------------------------------------------------------- #
def _build_qnn(threshold: float = 0.5) -> EstimatorQNN:
    """Build and return a QCNNGen182 EstimatorQNN.

    The ansatz consists of three convolutional and pooling layers.
    The feature map is a hybrid of a quantum filter and a ZFeatureMap.
    """
    algorithm_globals.random_seed = 12345
    estimator = Estimator()

    # Feature map
    feature_map = _custom_feature_map(num_qubits=8, threshold=threshold)

    # Ansatz
    ansatz = QuantumCircuit(8, name="Ansatz")
    # First Convolution + Pooling
    ansatz.compose(_conv_layer(8, "c1"), inplace=True)
    ansatz.compose(_pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), inplace=True)
    # Second Convolution + Pooling
    ansatz.compose(_conv_layer(4, "c2"), inplace=True)
    ansatz.compose(_pool_layer([0, 1], [2, 3], "p2"), inplace=True)
    # Third Convolution + Pooling
    ansatz.compose(_conv_layer(2, "c3"), inplace=True)
    ansatz.compose(_pool_layer([0], [1], "p3"), inplace=True)

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

# --------------------------------------------------------------------------- #
#  Quantum QCNNGen182 wrapper class
# --------------------------------------------------------------------------- #
class QCNNGen182:
    """Quantum QCNNGen182 wrapper.

    Parameters
    ----------
    threshold : float, default 0.5
        Threshold used in the quantum filter.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.qnn = _build_qnn(threshold=self.threshold)

    def __call__(self) -> EstimatorQNN:
        """Return the underlying EstimatorQNN."""
        return self.qnn

__all__ = ["QCNNGen182"]
