"""Quantum implementation of the hybrid QCNN.

The `HybridQCNN` function constructs a quantum neural network that
combines an autoencoder sub‑circuit with the convolution‑pooling
structure of the QCNN seed.  The autoencoder prepares a latent
representation that is then processed by successive convolutional
and pooling layers.  The design mirrors the classical `HybridQCNN`
model but operates on quantum states, enabling direct comparison.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals


def HybridQCNN(input_dim: int, num_latent: int = 3, num_trash: int = 2) -> EstimatorQNN:
    """Build a quantum neural network that fuses an autoencoder and QCNN layers.

    Parameters
    ----------
    input_dim : int
        Number of classical input features.
    num_latent : int
        Size of the autoencoder latent space.
    num_trash : int
        Number of auxiliary qubits used during encoding.
    """
    algorithm_globals.random_seed = 12345

    # --- Autoencoder sub‑circuit -----------------------------------------
    def autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)

        # Variational ansatz for the latent + trash qubits
        circuit.append(
            RealAmplitudes(num_latent + num_trash, reps=5),
            range(0, num_latent + num_trash),
        )
        circuit.barrier()

        # Swap‑test style entanglement with a single auxiliary qubit
        aux = num_latent + 2 * num_trash
        circuit.h(aux)
        for i in range(num_trash):
            circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
        circuit.h(aux)
        circuit.measure(aux, cr[0])

        return circuit

    # --- Convolution & Pooling layers ------------------------------------
    def conv_circuit(params: ParameterVector) -> QuantumCircuit:
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

    def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc.compose(conv_circuit(params[param_index: param_index + 3]), [q1, q2], inplace=True)
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc.compose(conv_circuit(params[param_index: param_index + 3]), [q1, q2], inplace=True)
            qc.barrier()
            param_index += 3
        return qc

    def pool_circuit(params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for i, (src, snk) in enumerate(zip(sources, sinks)):
            qc.compose(pool_circuit(params[i * 3: i * 3 + 3]), [src, snk], inplace=True)
            qc.barrier()
        return qc

    # --- Assemble the full ansatz ----------------------------------------
    feature_map = ZFeatureMap(input_dim)
    ansatz = QuantumCircuit(input_dim)

    # Autoencoder sub‑circuit (executed on the first few qubits)
    ae_circ = autoencoder_circuit(num_latent, num_trash)
    # Embed the autoencoder on the first (num_latent + 2*num_trash + 1) qubits
    ansatz.compose(ae_circ, range(num_latent + 2 * num_trash + 1), inplace=True)

    # First QCNN convolution + pooling
    conv1 = conv_layer(input_dim, "c1")
    pool1 = pool_layer(list(range(0, input_dim, 2)), list(range(1, input_dim, 2)), "p1")
    ansatz.compose(conv1, range(input_dim), inplace=True)
    ansatz.compose(pool1, range(input_dim), inplace=True)

    # Second QCNN convolution + pooling on the reduced qubits
    reduced = input_dim // 2
    conv2 = conv_layer(reduced, "c2")
    pool2 = pool_layer(
        list(range(0, reduced, 2)), list(range(1, reduced, 2)), "p2"
    )
    ansatz.compose(conv2, range(reduced), inplace=True)
    ansatz.compose(pool2, range(reduced), inplace=True)

    # Third QCNN convolution + pooling
    reduced2 = reduced // 2
    conv3 = conv_layer(reduced2, "c3")
    pool3 = pool_layer([0], [1], "p3")
    ansatz.compose(conv3, range(reduced2), inplace=True)
    ansatz.compose(pool3, range(reduced2), inplace=True)

    # Combine feature map and ansatz
    full_circ = QuantumCircuit(input_dim)
    full_circ.compose(feature_map, range(input_dim), inplace=True)
    full_circ.compose(ansatz, range(input_dim), inplace=True)

    # Observable for a single‑qubit read‑out
    observable = SparsePauliOp.from_list([("Z" + "I" * (input_dim - 1), 1)])

    estimator = Estimator()

    qnn = EstimatorQNN(
        circuit=full_circ.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn


__all__ = ["HybridQCNN"]
