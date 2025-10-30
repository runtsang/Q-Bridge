import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from typing import Iterable, Tuple, List

# --------------------------------------------------------------------------- #
# QCNN ansatz primitives (adapted from the reference QCNN)
# --------------------------------------------------------------------------- #
def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Elementary 2‑qubit convolution gate used inside the QCNN ansatz."""
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

def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Convolutional layer that stitches together the elementary 2‑qubit blocks."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc

def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Elementary 2‑qubit pooling gate used inside the QCNN ansatz."""
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    return target

def pool_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Pooling layer that stitches together the elementary 2‑qubit blocks."""
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        qc = qc.compose(pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc

# --------------------------------------------------------------------------- #
# Unified quantum classifier construction
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a data‑encoded QCNN ansatz:
    1. Encode the classical input with a ZFeatureMap.
    2. Apply a stack of *depth* convolution–pooling pairs.
    3. Measure a single Z observable on the first qubit.
    The interface matches the original seed for seamless integration.
    """
    # 1. Data encoding
    encoding = ParameterVector("x", num_qubits)

    circuit = QuantumCircuit(num_qubits)
    circuit.append(ZFeatureMap(num_qubits).to_instruction(), range(num_qubits))

    # 2. QCNN ansatz
    for i in range(depth):
        circuit.compose(conv_layer(num_qubits, f"c{i+1}"), range(num_qubits), inplace=True)
        circuit.compose(pool_layer(num_qubits, f"p{i+1}"), range(num_qubits), inplace=True)

    # 3. Observable
    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

    return circuit, list(encoding), list(circuit.parameters), [observable]

__all__ = ["build_classifier_circuit"]
