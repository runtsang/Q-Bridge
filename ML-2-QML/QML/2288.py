"""Quantum hybrid autoencoder using QCNN‑style encoding and a swap‑test decoder.

The circuit combines a QCNN feature map, convolutional and pooling layers,
and a variational decoder based on RealAmplitudes.  The output is
wrapped in a :class:`qiskit_machine_learning.neural_networks.SamplerQNN`
so it can be trained with a classical optimizer.
"""

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals


def _conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """A 2‑qubit convolution unit used in the QCNN stack."""
    qc = QuantumCircuit(num_qubits)
    qubits = list(range(num_qubits))
    for i in range(0, num_qubits, 2):
        qc.cx(qubits[i], qubits[i + 1])
        qc.rz(np.pi / 4, qubits[i])
        qc.ry(np.pi / 4, qubits[i + 1])
    return qc


def _pool_layer(sources: list[int], sinks: list[int]) -> QuantumCircuit:
    """A simple 2‑qubit pooling operation."""
    qc = QuantumCircuit(len(sources) + len(sinks))
    for src, snk in zip(sources, sinks):
        qc.cx(src, snk)
        qc.rz(np.pi / 2, snk)
    return qc


def Autoencoder() -> SamplerQNN:
    """Build a hybrid quantum autoencoder circuit."""
    algorithm_globals.random_seed = 42
    sampler = StatevectorSampler()

    # Feature map – encode classical data into qubit amplitudes
    feature_map = ZFeatureMap(num_qubits=8, reps=1)

    # Encoding stage – QCNN layers compress 8 qubits into 3 latent qubits
    conv1 = _conv_layer(8, "c1")
    pool1 = _pool_layer(list(range(4)), list(range(4, 6)))  # compress to 4
    conv2 = _conv_layer(4, "c2")
    pool2 = _pool_layer([0, 1], [2, 3])  # compress to 2
    conv3 = _conv_layer(2, "c3")
    # Final latent register (3 qubits) + 5 trash qubits + 1 auxiliary for swap test
    latent = 3
    trash = 2
    aux = 1
    total_qubits = latent + 2 * trash + aux

    # Build the full encoder circuit
    encoder = QuantumCircuit(total_qubits)
    encoder.compose(feature_map, range(8), inplace=True)
    encoder.compose(conv1, range(8), inplace=True)
    encoder.compose(pool1, range(8), inplace=True)
    encoder.compose(conv2, range(4, 8), inplace=True)
    encoder.compose(pool2, range(4, 8), inplace=True)
    encoder.compose(conv3, range(2, 4), inplace=True)

    # Decoder – swap‑test based reconstruction
    decoder = QuantumCircuit(total_qubits)
    # Swap‑test ancilla
    ancilla = latent + 2 * trash
    decoder.h(ancilla)
    for i in range(trash):
        decoder.cswap(ancilla, latent + i, latent + trash + i)
    decoder.h(ancilla)
    decoder.measure(ancilla, 0)

    # Variational decoder ansatz on latent qubits
    ansatz = RealAmplitudes(num_qubits=latent, reps=3)
    decoder.compose(ansatz, range(latent), inplace=True)

    # Assemble full autoencoder circuit
    qc = QuantumCircuit(total_qubits, 1)
    qc.compose(encoder, range(total_qubits), inplace=True)
    qc.compose(decoder, range(total_qubits), inplace=True)

    def identity_interpret(x):
        return x

    qnn = SamplerQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        interpret=identity_interpret,
        output_shape=1,
        sampler=sampler,
    )
    return qnn
