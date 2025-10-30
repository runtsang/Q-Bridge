"""
Quantum implementation of QCNNGen215 using Qiskit.

The circuit is built from two primitive modules:

* `conv_circuit` – a 2‑qubit variational block that mimics the quantum
  convolution step.
* `pool_circuit` – a 2‑qubit block that reduces the effective qubit count
  by discarding one qubit after a controlled operation.

Higher‑level helpers (`conv_layer`, `pool_layer`) stitch these primitives
into a QCNN‑style architecture.  The final circuit is combined with a
ZFeatureMap and wrapped into an EstimatorQNN so that it can be trained
with classical optimizers.

The quantum model is fully compatible with the classical `QCNNGen215`
in terms of input dimensionality and output shape, enabling direct
hybrid experiments.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals
from typing import List

# --------------------------------------------------------------------------- #
#  Core building blocks – convolution and pooling
# --------------------------------------------------------------------------- #
def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """2‑qubit convolution primitive with 3 trainable parameters."""
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


def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """2‑qubit pooling primitive with 3 trainable parameters."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


# --------------------------------------------------------------------------- #
#  Layer helpers – apply primitives across many qubits
# --------------------------------------------------------------------------- #
def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Builds a convolutional layer that pairs adjacent qubits."""
    qc = QuantumCircuit(num_qubits)
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        sub = conv_circuit(params[param_index:param_index + 3])
        qc.append(sub.to_instruction(), [q1, q2])
        qc.barrier()
        param_index += 3
    return qc


def pool_layer(sources: List[int], sinks: List[int], param_prefix: str) -> QuantumCircuit:
    """Builds a pooling layer that maps source qubits to sink qubits."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    param_index = 0
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    for src, sink in zip(sources, sinks):
        sub = pool_circuit(params[param_index:param_index + 3])
        qc.append(sub.to_instruction(), [src, sink])
        qc.barrier()
        param_index += 3
    return qc


# --------------------------------------------------------------------------- #
#  QCNN ansatz – stacked conv + pool layers
# --------------------------------------------------------------------------- #
def build_qcnn_ansatz(num_qubits: int) -> QuantumCircuit:
    """Constructs a QCNN ansatz mirroring the classical depth."""
    qc = QuantumCircuit(num_qubits)

    # Layer 1: conv over 8 qubits
    qc.append(conv_layer(num_qubits, "c1"), range(num_qubits))

    # Layer 2: pool 8→4
    qc.append(pool_layer(list(range(0, 8, 2)), list(range(4, 8)), "p1"), range(num_qubits))

    # Layer 3: conv over 4 qubits
    qc.append(conv_layer(4, "c2"), list(range(4, 8)))

    # Layer 4: pool 4→2
    qc.append(pool_layer([0, 1, 2, 3], [4, 5], "p2"), range(8))

    # Layer 5: conv over 2 qubits
    qc.append(conv_layer(2, "c3"), list(range(6, 8)))

    # Layer 6: pool 2→1
    qc.append(pool_layer([0], [1], "p3"), range(8))

    return qc


# --------------------------------------------------------------------------- #
#  Final QCNN model – feature map + ansatz + readout
# --------------------------------------------------------------------------- #
def QCNNGen215QNN(num_qubits: int = 8) -> EstimatorQNN:
    """
    Returns a Qiskit EstimatorQNN that implements the QCNNGen215 architecture.

    Parameters
    ----------
    num_qubits : int
        Number of input qubits (must match the feature map size).

    Returns
    -------
    EstimatorQNN
        Quantum neural network ready for training with a classical optimizer.
    """
    algorithm_globals.random_seed = 12345
    estimator = StatevectorEstimator()

    # Feature map – encode classical data into a quantum state
    feature_map = ZFeatureMap(num_qubits)
    feature_map_params = feature_map.parameters

    # Ansatz – the QCNN circuit
    ansatz = build_qcnn_ansatz(num_qubits)

    # Observable – measure Z on the last qubit
    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

    qnn = EstimatorQNN(
        circuit=ansatz.decompose(),
        observables=observable,
        input_params=feature_map_params,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = ["QCNNGen215QNN"]
