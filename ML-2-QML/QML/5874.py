import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

def _conv_circuit(params):
    """Two‑qubit convolution unitary used in the QCNN."""
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

def _pool_circuit(params):
    """Two‑qubit pooling unitary used in the QCNN."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def _conv_layer(num_qubits, param_prefix):
    """Build a convolution layer that acts on adjacent qubit pairs."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for i in range(0, num_qubits, 2):
        sub = _conv_circuit(params[i // 2 * 3:(i // 2 + 1) * 3])
        qc.append(sub, [i, i + 1])
        qc.barrier()
    return qc

def _pool_layer(sources, sinks, param_prefix):
    """Build a pooling layer that maps a source qubit to a sink qubit."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    for idx, (src, snk) in enumerate(zip(sources, sinks)):
        sub = _pool_circuit(params[idx * 3:(idx + 1) * 3])
        qc.append(sub, [src, snk])
        qc.barrier()
    return qc

def QCNN(num_layers: int = 3, seed: int | None = None) -> EstimatorQNN:
    """
    Factory that builds a QCNN EstimatorQNN with a depth‑controlled
    convolutional stack.  The architecture closely mirrors the
    original QCNN but exposes the number of layers as a parameter.
    """
    if seed is not None:
        np.random.seed(seed)

    feature_map = ZFeatureMap(8)
    ansatz = QuantumCircuit(8, name="Ansatz")

    if num_layers >= 1:
        ansatz.compose(_conv_layer(8, "c1"), range(8), inplace=True)
        ansatz.compose(_pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), range(8), inplace=True)

    if num_layers >= 2:
        ansatz.compose(_conv_layer(4, "c2"), range(4, 8), inplace=True)
        ansatz.compose(_pool_layer([0, 1], [2, 3], "p2"), range(4, 8), inplace=True)

    if num_layers >= 3:
        ansatz.compose(_conv_layer(2, "c3"), range(6, 8), inplace=True)
        ansatz.compose(_pool_layer([0], [1], "p3"), range(6, 8), inplace=True)

    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    estimator = Estimator()
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
        gradient_method="parameter-shift",
    )
    return qnn
