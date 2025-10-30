import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator
from typing import Tuple, Iterable

# ----- QCNN ansatz helpers -----
def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Single convolutional unit used in QCNN."""
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
    """Pooling unit used in QCNN."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Build a convolutional layer over `num_qubits` qubits."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        qc.append(conv_circuit(params[idx:idx+3]), [q1, q2])
        qc.barrier()
        idx += 3
    for q1, q2 in zip(range(1, num_qubits, 2), range(2, num_qubits, 2)):
        qc.append(conv_circuit(params[idx:idx+3]), [q1, q2])
        qc.barrier()
        idx += 3
    return qc

def pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    """Build a pooling layer over selected qubits."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    for src, sink, idx in zip(sources, sinks, range(len(sources))):
        qc.append(pool_circuit(params[idx:idx+3]), [src, sink])
        qc.barrier()
    return qc

def build_qcnn_ansatz(num_qubits: int, depth: int) -> QuantumCircuit:
    """Full QCNN ansatz consisting of alternating conv and pool layers."""
    ansatz = QuantumCircuit(num_qubits)
    # First convolution
    ansatz.compose(conv_layer(num_qubits, "c1"), inplace=True)
    # First pooling
    ansatz.compose(pool_layer(list(range(num_qubits//2)), list(range(num_qubits//2, num_qubits)), "p1"), inplace=True)
    # Second convolution
    ansatz.compose(conv_layer(num_qubits//2, "c2"), inplace=True)
    # Second pooling
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), inplace=True)
    # Third convolution
    ansatz.compose(conv_layer(num_qubits//4, "c3"), inplace=True)
    # Third pooling
    ansatz.compose(pool_layer([0], [1], "p3"), inplace=True)
    return ansatz

# ----- Self‑attention circuit helper -----
def attention_circuit(rotation_params: ParameterVector, entangle_params: ParameterVector) -> QuantumCircuit:
    """Build a self‑attention style circuit for a block of qubits."""
    n = len(rotation_params) // 3
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.rx(rotation_params[3*i], i)
        qc.ry(rotation_params[3*i+1], i)
        qc.rz(rotation_params[3*i+2], i)
    for i in range(n-1):
        qc.crx(entangle_params[i], i, i+1)
    return qc

# ----- Full hybrid circuit -----
def build_full_circuit(num_qubits: int, qcnn_depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """Combine encoding, QCNN, attention, and measurement into a single parametric circuit."""
    # Input encoding (2 parameters per qubit)
    input_params = ParameterVector("x", 2 * num_qubits)

    # QCNN variational parameters
    qcnn_params = ParameterVector("qcnn", num_qubits * qcnn_depth)

    # Attention parameters
    attn_rot_params = ParameterVector("rot", 3 * num_qubits)
    attn_ent_params = ParameterVector("ent", num_qubits - 1)

    # Build the circuit
    circuit = QuantumCircuit(num_qubits)

    # Encoding
    for i in range(num_qubits):
        circuit.rx(input_params[2*i], i)
        circuit.ry(input_params[2*i+1], i)

    # QCNN ansatz
    qc_ansatz = build_qcnn_ansatz(num_qubits, qcnn_depth)
    circuit.compose(qc_ansatz, inplace=True)

    # Self‑attention block
    attn_circ = attention_circuit(attn_rot_params, attn_ent_params)
    circuit.compose(attn_circ, inplace=True)

    # Observables: use a Z on the last qubit as a simple binary classifier
    observables = [SparsePauliOp("Z" + "I" * (num_qubits - 1))]

    return circuit, input_params, [qcnn_params, attn_rot_params, attn_ent_params], observables

def SamplerQNN() -> EstimatorQNN:
    """Factory returning a hybrid quantum neural network."""
    num_qubits = 4
    qcnn_depth = 3
    circuit, input_params, weight_params, observables = build_full_circuit(num_qubits, qcnn_depth)
    estimator = Estimator()
    return EstimatorQNN(circuit=circuit,
                        observables=observables,
                        input_params=input_params,
                        weight_params=weight_params,
                        estimator=estimator)

__all__ = ["SamplerQNN"]
