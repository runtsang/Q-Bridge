"""Quantum implementation of the hybrid QCNN with a fully‑connected layer."""
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

algorithm_globals.random_seed = 12345
estimator = StatevectorEstimator()

def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
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

def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def _conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits // 2 * 3)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        sub = _conv_circuit(params[idx:idx+3])
        qc.append(sub, [q1, q2])
        qc.barrier()
        idx += 3
    return qc

def _pool_layer(sources: Iterable[int], sinks: Iterable[int], prefix: str) -> QuantumCircuit:
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=len(sources) * 3)
    idx = 0
    for src, sink in zip(sources, sinks):
        sub = _pool_circuit(params[idx:idx+3])
        qc.append(sub, [src, sink])
        qc.barrier()
        idx += 3
    return qc

def _full_connected_layer(params: ParameterVector) -> QuantumCircuit:
    """Single‑qubit fully‑connected layer mimicking the classical FCL."""
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.barrier()
    qc.ry(params[0], 0)
    qc.measure_all()
    return qc

def QCNN() -> EstimatorQNN:
    """Build the full QCNN ansatz with an additional fully‑connected layer."""
    # Feature map (8‑qubit Z‑feature map)
    feature_map = QuantumCircuit(8)
    for i in range(8):
        feature_map.h(i)
        feature_map.rz(0.0, i)  # placeholder for input encoding

    # Ansatz construction
    ansatz = QuantumCircuit(8)
    # Convolution & pooling stages
    ansatz.append(_conv_layer(8, "c1"), range(8))
    ansatz.append(_pool_layer(list(range(4)), list(range(4, 8)), "p1"), range(8))
    ansatz.append(_conv_layer(4, "c2"), range(4, 8))
    ansatz.append(_pool_layer([0, 1], [2, 3], "p2"), range(4, 8))
    ansatz.append(_conv_layer(2, "c3"), range(6, 8))
    ansatz.append(_pool_layer([0], [1], "p3"), range(6, 8))

    # Add fully‑connected layer on a single qubit (the last qubit)
    fc_params = ParameterVector("fc", length=1)
    fc_layer = _full_connected_layer(fc_params)
    ansatz.append(fc_layer, [7])

    # Full circuit: feature map + ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # Construct EstimatorQNN
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn
