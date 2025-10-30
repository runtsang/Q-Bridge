"""Quantum hybrid QCNN + autoencoder circuit.

The quantum implementation mirrors the classical hybrid:
1. A feature map encodes the input data.
2. Convolution and pooling layers (adapted from the QCNN seed) process
   the qubits.
3. An autoencoder ansatz (swap‑test style) reduces the state to a
   low‑dimensional latent subspace.
4. A sampler is used to obtain measurement probabilities, which are
   interpreted as a binary classification output.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals


def QCNNAutoEncoderQNN(
    num_features: int,
    num_latent: int = 3,
    num_trash: int = 2,
    ansatz_reps: int = 5,
) -> SamplerQNN:
    """Builds a quantum circuit that combines QCNN layers with a swap‑test
    autoencoder ansatz and returns a :class:`SamplerQNN` for training.
    """
    algorithm_globals.random_seed = 42
    sampler = Sampler()

    # ----- Feature map -----
    feature_map = ZFeatureMap(num_features)

    # ----- Convolution helper -----
    def conv_circuit(params):
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

    def conv_layer(num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits, name="Convolution")
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for idx, (q1, q2) in enumerate(zip(range(0, num_qubits, 2), range(1, num_qubits, 2))):
            qc.append(conv_circuit(params[3*idx:3*idx+3]), [q1, q2])
            qc.barrier()
        return qc

    # ----- Pooling helper -----
    def pool_circuit(params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def pool_layer(sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling")
        params = ParameterVector(param_prefix, length=len(sources) * 3)
        for idx, (s, t) in enumerate(zip(sources, sinks)):
            qc.append(pool_circuit(params[3*idx:3*idx+3]), [s, t])
            qc.barrier()
        return qc

    # ----- Autoencoder ansatz (swap‑test) -----
    def auto_encoder_circuit(num_latent, num_trash):
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        qc.compose(
            RealAmplitudes(num_latent + num_trash, reps=ansatz_reps),
            range(num_latent + num_trash),
            inplace=True,
        )
        qc.barrier()
        aux = num_latent + 2 * num_trash
        qc.h(aux)
        for i in range(num_trash):
            qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    # ----- Assemble full circuit -----
    qc = QuantumCircuit(num_features)
    qc.compose(feature_map, range(num_features), inplace=True)

    # First convolution + pooling
    qc.compose(conv_layer(num_features, "c1"), range(num_features), inplace=True)
    qc.compose(
        pool_layer(
            list(range(num_features // 2)),
            list(range(num_features // 2, num_features)),
            "p1",
        ),
        range(num_features),
        inplace=True,
    )

    # Autoencoder layer
    ae_circuit = auto_encoder_circuit(num_latent, num_trash)
    qc.compose(ae_circuit, range(num_features), inplace=True)

    # Interpret measurement outcomes as a single binary output
    def interpret(x):
        # x[0] = prob(|0>), x[1] = prob(|1>)
        return np.array([x[1]])

    qnn = SamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=ae_circuit.parameters,
        interpret=interpret,
        output_shape=1,
        sampler=sampler,
    )
    return qnn


__all__ = ["QCNNAutoEncoderQNN"]
