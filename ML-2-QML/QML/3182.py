import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.quantum_info import SparsePauliOp

def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """A single QCNN convolution block used in the ansatz."""
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

def _conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Build a QCNN convolutional layer across all qubit pairs."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        block = _conv_circuit(params[param_index:param_index+3])
        qc.append(block, [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        block = _conv_circuit(params[param_index:param_index+3])
        qc.append(block, [q1, q2])
        qc.barrier()
        param_index += 3
    return qc

def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """A QCNN pooling block that reduces entanglement."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def _pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    """Build a QCNN pooling layer mapping sources to sinks."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for src, snk in zip(sources, sinks):
        block = _pool_circuit(params[param_index:param_index+3])
        qc.append(block, [src, snk])
        qc.barrier()
        param_index += 3
    return qc

def HybridAutoencoder() -> EstimatorQNN:
    """Quantum hybrid autoencoder implementing a QCNN ansatz."""
    algorithm_globals.random_seed = 42
    estimator = Estimator()

    num_qubits = 8  # latent dimensionality
    feature_map = ZFeatureMap(num_qubits, reps=1)

    # Build the QCNN ansatz
    ansatz = QuantumCircuit(num_qubits, name="QCNN Ansatz")
    # First convolution + pooling
    ansatz.compose(_conv_layer(num_qubits, "c1"), inplace=True)
    ansatz.compose(_pool_layer(list(range(num_qubits // 2)), list(range(num_qubits // 2, num_qubits)), "p1"), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(num_qubits, name="QCNN Autoencoder")
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    # Observables to read out a latent vector of size num_qubits
    observables = [SparsePauliOp.from_list([("Z" + "I" * (num_qubits-1-i), 1)]) for i in range(num_qubits)]

    qnn = EstimatorQNN(
        circuit=circuit,
        observables=observables,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = ["HybridAutoencoder"]
