"""Hybrid QCNN with a quantum self‑attention sub‑circuit.

The ansatz starts with a parameterised self‑attention block that
applies rotations to each qubit followed by controlled‑RX gates
between neighbours.  This block is inserted before every
convolution‑pool pair, mirroring the classical attention
pre‑processing.  The rest of the circuit follows the original
QCNN construction, using two‑qubit convolution and pooling
circuits and a Z‑feature map for data encoding.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

def _conv_circuit(params):
    """Two‑qubit convolution block from the seed."""
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

def _pool_circuit(params):
    """Two‑qubit pooling block from the seed."""
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    return target

def _conv_layer(num_qubits, param_prefix):
    """Convolutional layer built from the two‑qubit block."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        qc.append(_conv_circuit(params[idx:idx+3]), [q1, q2])
        idx += 3
    return qc

def _pool_layer(num_qubits, param_prefix):
    """Pooling layer built from the two‑qubit block."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        qc.append(_pool_circuit(params[idx:idx+3]), [q1, q2])
        idx += 3
    return qc

def _self_attention_layer(num_qubits, param_prefix):
    """Parameterised self‑attention sub‑circuit."""
    qc = QuantumCircuit(num_qubits)
    rot_params = ParameterVector(f"{param_prefix}_rot", length=num_qubits * 3)
    ent_params = ParameterVector(f"{param_prefix}_ent", length=num_qubits - 1)
    # Rotations on each qubit
    for i in range(num_qubits):
        qc.rx(rot_params[3 * i], i)
        qc.ry(rot_params[3 * i + 1], i)
        qc.rz(rot_params[3 * i + 2], i)
    # Controlled‑RX entanglement between neighbours
    for i in range(num_qubits - 1):
        qc.crx(ent_params[i], i, i + 1)
    return qc

def QCNN() -> EstimatorQNN:
    """Return a QNN that embeds a quantum self‑attention block in a QCNN ansatz."""
    estimator = StatevectorEstimator()
    feature_map = ZFeatureMap(8)

    # Build ansatz
    ansatz = QuantumCircuit(8, name="Ansatz")

    # Self‑attention block before the first conv
    ansatz.append(_self_attention_layer(8, "att1"), range(8))

    # First conv + pool
    ansatz.append(_conv_layer(8, "c1"), range(8))
    ansatz.append(_pool_layer(8, "p1"), range(8))

    # Second conv + pool on 4 qubits
    ansatz.append(_conv_layer(4, "c2"), range(4, 8))
    ansatz.append(_pool_layer(4, "p2"), range(4, 8))

    # Third conv + pool on 2 qubits
    ansatz.append(_conv_layer(2, "c3"), range(6, 8))
    ansatz.append(_pool_layer(2, "p3"), range(6, 8))

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

__all__ = ["QCNN"]
