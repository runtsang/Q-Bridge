"""Quantum Hybrid Autoencoder that stitches a QCNN feature map, convolution, pooling,
and an autoencoder ansatz into a variational circuit.

The circuit uses a ZFeatureMap for data embedding, followed by multiple
convolutional and pooling layers that emulate the QCNN structure.  Finally,
a RealAmplitudes ansatz implements the encoder–decoder mapping and
a swap-test-based readout yields a two-bit measurement that can be
interpreted as the latent representation.
"""

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals

algorithm_globals.random_seed = 42


def _conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Builds a convolutional layer as a block of 2‑qubit conv circuits."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        sub = _single_conv(params[i * 3 : (i + 2) * 3])
        qc.append(sub, [i, i + 1])
    return qc


def _single_conv(params):
    """Single 2‑qubit convolution circuit."""
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


def _pool_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Pooling layer that compresses qubit pairs."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for i in range(0, num_qubits, 2):
        sub = _single_pool(params[i // 2 * 3 : (i // 2 + 1) * 3])
        qc.append(sub, [i, i + 1])
    return qc


def _single_pool(params):
    """Single 2‑qubit pooling circuit."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def HybridAutoencoderQNN() -> SamplerQNN:
    """Constructs a variational circuit that combines QCNN layers and an autoencoder ansatz."""
    # Data embedding
    feature_map = ZFeatureMap(8)
    # Convolution–pooling stack
    conv1 = _conv_layer(8, "c1")
    pool1 = _pool_layer(8, "p1")
    conv2 = _conv_layer(4, "c2")
    pool2 = _pool_layer(4, "p2")
    conv3 = _conv_layer(2, "c3")
    pool3 = _pool_layer(2, "p3")

    # Autoencoder ansatz on the remaining qubits
    ae_ansatz = RealAmplitudes(4, reps=5)

    # Assemble the full circuit
    qc = QuantumCircuit(8)
    qc.compose(feature_map, range(8), inplace=True)
    qc.compose(conv1, range(8), inplace=True)
    qc.compose(pool1, range(8), inplace=True)
    qc.compose(conv2, range(4, 8), inplace=True)
    qc.compose(pool2, range(4, 8), inplace=True)
    qc.compose(conv3, range(6, 8), inplace=True)
    qc.compose(pool3, range(6, 8), inplace=True)
    qc.compose(ae_ansatz, range(4, 8), inplace=True)

    # Readout: swap‑test style measurement on a dedicated ancilla
    ancilla = QuantumRegister(1, "anc")
    qc.add_register(ancilla)
    qc.h(ancilla[0])
    for i in range(4):
        qc.cswap(ancilla[0], 4 + i, i)
    qc.h(ancilla[0])
    cr = ClassicalRegister(1, "c")
    qc.add_register(cr)
    qc.measure(ancilla[0], cr[0])

    sampler = Sampler()
    # Interpret the 2‑bit outcome as a latent vector
    def interpret(x):
        return np.array([float(x[0]), float(x[1])])

    qnn = SamplerQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ae_ansatz.parameters,
        interpret=interpret,
        output_shape=2,
        sampler=sampler,
    )
    return qnn


__all__ = ["HybridAutoencoderQNN"]
