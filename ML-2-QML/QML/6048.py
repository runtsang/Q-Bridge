import json
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

def _conv_circuit(params):
    """Two‑qubit convolution unit used in QCNN."""
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
    """Two‑qubit pooling unit used in QCNN."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def _conv_layer(num_qubits, param_prefix):
    """Builds a convolutional layer by pairing qubits."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    qubits = list(range(num_qubits))
    for i in range(0, num_qubits, 2):
        qc.compose(_conv_circuit(params[3*i:3*(i+1)]), [qubits[i], qubits[i+1]], inplace=True)
        qc.barrier()
    return qc

def _pool_layer(sources, sinks, param_prefix):
    """Builds a pooling layer by pairing source and sink qubits."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for idx, (src, snk) in enumerate(zip(sources, sinks)):
        qc.compose(_pool_circuit(params[3*idx:3*(idx+1)]), [src, snk], inplace=True)
        qc.barrier()
    return qc

def HybridAutoencoder() -> EstimatorQNN:
    """Quantum autoencoder that combines a QCNN ansatz with a swap‑test style reconstruction."""
    algorithm_globals.random_seed = 42
    estimator = Estimator()
    feature_map = ZFeatureMap(8)
    ansatz = QuantumCircuit(8, name="QCNNAnsatz")

    # First convolution + pooling
    ansatz.compose(_conv_layer(8, "c1"), list(range(8)), inplace=True)
    ansatz.compose(_pool_layer([0,1,2,3],[4,5,6,7],"p1"), list(range(8)), inplace=True)

    # Second convolution + pooling
    ansatz.compose(_conv_layer(4, "c2"), list(range(4,8)), inplace=True)
    ansatz.compose(_pool_layer([0,1],[2,3],"p2"), list(range(4,8)), inplace=True)

    # Third convolution + pooling
    ansatz.compose(_conv_layer(2, "c3"), list(range(6,8)), inplace=True)
    ansatz.compose(_pool_layer([0],[1],"p3"), list(range(6,8)), inplace=True)

    # Ancilla for swap‑test reconstruction
    ancilla = QuantumRegister(1, "anc")
    qc = QuantumCircuit(8, 1, name="AutoencoderCircuit")
    qc.add_register(ancilla)

    # Apply feature map and ansatz
    qc.compose(feature_map, range(8), inplace=True)
    qc.compose(ansatz, range(8), inplace=True)

    # Swap‑test between first three latent qubits and the corresponding input qubits
    qc.h(ancilla[0])
    for i in range(3):
        qc.cswap(ancilla[0], i, i)
    qc.h(ancilla[0])

    # Observable on the ancilla captures reconstruction fidelity
    observable = SparsePauliOp.from_list([("Z", 1)])

    qnn = EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = ["HybridAutoencoder"]
