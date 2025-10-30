"""Quantum circuits for the hybrid autoencoder and QCNN classifier.

The QML module supplies the quantum latent circuit used by the classical
HybridAutoEncoderNet and a QCNN‑style classifier that can be wrapped by
EstimatorQNN or SamplerQNN for hybrid training.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector, Parameter
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

# --------------------------------------------------------------------------- #
# Utility: quantum autoencoder circuit
# --------------------------------------------------------------------------- #
def quantum_autoencoder_circuit(num_qubits: int, num_trash: int) -> QuantumCircuit:
    """Build a swap‑test based quantum autoencoder.

    Parameters
    ----------
    num_qubits : int
        Total qubits, including auxiliary qubit.
    num_trash : int
        Number of trash qubits used for encoding the latent space.
    """
    qr = QuantumRegister(num_qubits, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Encoder ansatz
    ansatz = RealAmplitudes(num_qubits - 1, reps=5)
    qc.compose(ansatz, range(num_qubits - 1), inplace=True)

    # Swap test to extract latent amplitudes
    aux = num_qubits - 1
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, i, num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])
    return qc

# --------------------------------------------------------------------------- #
# Wrapper for SamplerQNN
# --------------------------------------------------------------------------- #
def hybrid_autoencoder_qnn(num_qubits: int, num_trash: int) -> SamplerQNN:
    """Return a SamplerQNN that implements the autoencoder circuit."""
    algorithm_globals.random_seed = 42
    sampler = Sampler()
    circuit = quantum_autoencoder_circuit(num_qubits, num_trash)
    return SamplerQNN(
        circuit=circuit,
        input_params=[],
        weight_params=circuit.parameters,
        interpret=lambda x: x,
        output_shape=2,
        sampler=sampler,
    )

# --------------------------------------------------------------------------- #
# QCNN primitives
# --------------------------------------------------------------------------- #
def conv_circuit(params: ParameterVector, q1: int, q2: int) -> QuantumCircuit:
    """Convolution sub‑circuit used in QCNN layers."""
    c = QuantumCircuit(2)
    c.rz(-np.pi / 2, 1)
    c.cx(1, 0)
    c.rz(params[0], 0)
    c.ry(params[1], 1)
    c.cx(0, 1)
    c.ry(params[2], 1)
    c.cx(1, 0)
    c.rz(np.pi / 2, 0)
    return c

def pool_circuit(params: ParameterVector, q1: int, q2: int) -> QuantumCircuit:
    """Pooling sub‑circuit used in QCNN layers."""
    c = QuantumCircuit(2)
    c.rz(-np.pi / 2, 1)
    c.cx(1, 0)
    c.rz(params[0], 0)
    c.ry(params[1], 1)
    c.cx(0, 1)
    c.ry(params[2], 1)
    return c

def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Convolutional layer that applies conv_circuit to each adjacent pair."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        sub = conv_circuit(params[idx:idx+3], q1, q2)
        qc.append(sub, [q1, q2])
        idx += 3
    return qc

def pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    """Pooling layer that applies pool_circuit to each source–sink pair."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    idx = 0
    for source, sink in zip(sources, sinks):
        sub = pool_circuit(params[idx:idx+3], source, sink)
        qc.append(sub, [source, sink])
        idx += 3
    return qc

# --------------------------------------------------------------------------- #
# QCNN classifier circuit
# --------------------------------------------------------------------------- #
def qcnn_classifier_circuit() -> EstimatorQNN:
    """Return an EstimatorQNN that implements a QCNN classifier on 8 qubits."""
    algorithm_globals.random_seed = 12345

    # Feature map
    feature_map = ZFeatureMap(8)

    # Ansatz construction
    ansatz = QuantumCircuit(8)

    # First convolutional layer
    ansatz.compose(conv_layer(8, "c1"), range(8), inplace=True)

    # First pooling layer
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), range(8), inplace=True)

    # Second convolutional layer
    ansatz.compose(conv_layer(4, "c2"), range(4, 8), inplace=True)

    # Second pooling layer
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), range(4, 8), inplace=True)

    # Third convolutional layer
    ansatz.compose(conv_layer(2, "c3"), range(6, 8), inplace=True)

    # Third pooling layer
    ansatz.compose(pool_layer([0], [1], "p3"), range(6, 8), inplace=True)

    # Combine feature map and ansatz
    ansatz.compose(feature_map, range(8), inplace=True)

    # Observable for a binary classification
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    return EstimatorQNN(
        circuit=ansatz.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
    )

__all__ = [
    "quantum_autoencoder_circuit",
    "hybrid_autoencoder_qnn",
    "conv_layer",
    "pool_layer",
    "qcnn_classifier_circuit",
]
