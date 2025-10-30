import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals

def conv_circuit(params):
    """Two‑qubit parameterized block used in the QCNN ansatz."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi/2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(np.pi/2, 0)
    return qc

def pool_circuit(params):
    """Two‑qubit pooling block used in the QCNN ansatz."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi/2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def conv_layer(num_qubits, param_prefix):
    """Apply a convolutional layer over all adjacent qubit pairs."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        qc.append(conv_circuit(params[q1*3:q1*3+3]), [q1, q2])
    return qc

def pool_layer(sources, sinks, param_prefix):
    """Apply a pooling layer that maps a source qubit to a sink qubit."""
    qc = QuantumCircuit(len(sources) + len(sinks))
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    for src, snk in zip(sources, sinks):
        qc.append(pool_circuit(params[src*3:src*3+3]), [src, snk])
    return qc

def SamplerQNNGen327():
    """Quantum sampler that combines a QCNN‑style ansatz with a
    feature‑map and a graph‑based entanglement pattern."""
    algorithm_globals.random_seed = 12345

    # Feature map: 8‑qubit Z‑feature map
    feature_map = ZFeatureMap(8)

    # Build a QCNN‑style ansatz with 3 conv‑pool stages
    ansatz = QuantumCircuit(8)
    ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)
    ansatz.compose(pool_layer([0,1,2,3], [4,5,6,7], "p1"), list(range(8)), inplace=True)
    ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
    ansatz.compose(pool_layer([0,1], [2,3], "p2"), list(range(4, 8)), inplace=True)
    ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    # Observable: measure Z on the first qubit
    observable = SparsePauliOp.from_list([("Z" + "I"*7, 1)])

    # Sampler primitive
    sampler = StatevectorSampler()

    # Build the sampler QNN
    qnn = SamplerQNN(
        circuit=circuit,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        sampler=sampler
    )
    return qnn

__all__ = ["SamplerQNNGen327"]
