import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

def _clip(value: float, bound: float) -> float:
    """Clip a parameter to a symmetric bound."""
    return max(-bound, min(bound, value))

def conv_circuit(params, clip: bool = False) -> QuantumCircuit:
    """Two‑qubit convolution unit with optional parameter clipping."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(np.pi / 2, 0)
    if clip:
        for gate in qc.data:
            if gate[0].name in {"rz", "ry"}:
                gate[1].params[0] = _clip(gate[1].params[0], 5.0)
    return qc

def conv_layer(num_qubits: int, param_prefix: str, clip: bool = False) -> QuantumCircuit:
    """Convolutional layer composed of multiple conv_circuits."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc.append(conv_circuit(params[param_index:param_index+3], clip), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc.append(conv_circuit(params[param_index:param_index+3], clip), [q1, q2])
        qc.barrier()
        param_index += 3
    return qc

def pool_circuit(params, clip: bool = False) -> QuantumCircuit:
    """Two‑qubit pooling unit with optional clipping."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    if clip:
        for gate in qc.data:
            if gate[0].name in {"rz", "ry"}:
                gate[1].params[0] = _clip(gate[1].params[0], 5.0)
    return qc

def pool_layer(sources, sinks, param_prefix: str, clip: bool = False) -> QuantumCircuit:
    """Pooling layer connecting specified source and sink qubits."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc.append(pool_circuit(params[param_index:param_index+3], clip), [source, sink])
        qc.barrier()
        param_index += 3
    return qc

def QCNN__gen223_qnn() -> EstimatorQNN:
    """Builds a QCNN‑style EstimatorQNN with clipped parameters."""
    algorithm_globals.random_seed = 12345
    estimator = Estimator()

    feature_map = ZFeatureMap(8)
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    ansatz = QuantumCircuit(8, name="Ansatz")
    ansatz.compose(conv_layer(8, "c1", clip=True), list(range(8)), inplace=True)
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1", clip=True), list(range(8)), inplace=True)
    ansatz.compose(conv_layer(4, "c2", clip=True), list(range(4, 8)), inplace=True)
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2", clip=True), list(range(4, 8)), inplace=True)
    ansatz.compose(conv_layer(2, "c3", clip=True), list(range(6, 8)), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3", clip=True), list(range(6, 8)), inplace=True)

    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = ["QCNN__gen223_qnn"]
