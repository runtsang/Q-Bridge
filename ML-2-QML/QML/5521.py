from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap, RandomUnitary
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

# ----------------------------------------------------------------------
# Helper layers --------------------------------------------------------
# ----------------------------------------------------------------------
def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution block used in QCNN layers."""
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

def conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Stack of convolution blocks over the qubits."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits // 2 * 3)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        sub = conv_circuit(params[idx: idx + 3])
        qc.append(sub, [q1, q2])
        qc.barrier()
        idx += 3
    return qc

def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling block."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def pool_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Stack of pooling blocks over the qubits."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits // 2 * 3)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        sub = pool_circuit(params[idx: idx + 3])
        qc.append(sub, [q1, q2])
        qc.barrier()
        idx += 3
    return qc

def self_attention_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Parametric rotation + CRX entanglement – a quantum self‑attention block."""
    qc = QuantumCircuit(num_qubits)
    rot_params = ParameterVector(f"{prefix}_rot", length=num_qubits * 3)
    ent_params = ParameterVector(f"{prefix}_ent", length=num_qubits - 1)
    for i in range(num_qubits):
        qc.rx(rot_params[3 * i], i)
        qc.ry(rot_params[3 * i + 1], i)
        qc.rz(rot_params[3 * i + 2], i)
    for i in range(num_qubits - 1):
        qc.crx(ent_params[i], i, i + 1)
    return qc

def graph_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Random two‑qubit unitaries applied to each adjacent pair – encodes graph structure."""
    qc = QuantumCircuit(num_qubits)
    for i in range(0, num_qubits, 2):
        if i + 1 < num_qubits:
            # A fresh random unitary per pair
            rand_u = RandomUnitary(2, depth=1, seed=np.random.randint(0, 1e6))
            qc.append(rand_u, [i, i + 1])
    return qc

def quanvolution_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Two‑qubit random unitary per patch, mirroring a quanvolution filter."""
    qc = QuantumCircuit(num_qubits)
    for i in range(0, num_qubits, 2):
        if i + 1 < num_qubits:
            rand_u = RandomUnitary(2, depth=1, seed=np.random.randint(0, 1e6))
            qc.append(rand_u, [i, i + 1])
    return qc

# ----------------------------------------------------------------------
# Hybrid QCNN construction -----------------------------------------------
# ----------------------------------------------------------------------
def QCNN() -> EstimatorQNN:
    """
    Builds a variational quantum neural network that mirrors the classical
    HybridQCNN. The circuit consists of a feature map, followed by a stack of
    convolution, pooling, self‑attention, graph, and quanvolution layers.
    """
    # Feature map – the 8‑qubit ZFeatureMap encodes the input data
    feature_map = ZFeatureMap(8)
    feature_params = feature_map.parameters

    # Ansatz construction
    ansatz = QuantumCircuit(8, name="HybridAnsatz")

    # 1. First convolution + pooling
    ansatz.compose(conv_layer(8, "c1"), inplace=True)
    ansatz.compose(pool_layer(8, "p1"), inplace=True)

    # 2. Second convolution + pooling
    ansatz.compose(conv_layer(4, "c2"), inplace=True)
    ansatz.compose(pool_layer(4, "p2"), inplace=True)

    # 3. Third convolution + pooling
    ansatz.compose(conv_layer(2, "c3"), inplace=True)
    ansatz.compose(pool_layer(2, "p3"), inplace=True)

    # 4. Quantum self‑attention block
    ansatz.compose(self_attention_layer(8, "sa"), inplace=True)

    # 5. Graph‑based random unitaries
    ansatz.compose(graph_layer(8, "gr"), inplace=True)

    # 6. Quanvolution‑style random unitaries
    ansatz.compose(quanvolution_layer(8, "qv"), inplace=True)

    # The observable – a single‑qubit Z on the first qubit
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # Wrap in an EstimatorQNN
    estimator = Estimator()
    qnn = EstimatorQNN(
        circuit=ansatz.decompose(),
        observables=observable,
        input_params=feature_params,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn


__all__ = ["QCNN"]
