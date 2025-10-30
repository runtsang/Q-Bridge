"""Hybrid quantum neural network that merges QCNN layers with an autoencoder‑style entanglement block."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

# --------------------------------------------------------------------
# 1. QCNN building blocks
# --------------------------------------------------------------------
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

def conv_layer(num_qubits, prefix):
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits // 2 * 3)
    for i in range(0, num_qubits, 2):
        layer = conv_circuit(params[i // 2 * 3 : i // 2 * 3 + 3])
        qc.append(layer, [i, i + 1])
    return qc

def pool_circuit(params):
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def pool_layer(sources, sinks, prefix):
    num = len(sources) + len(sinks)
    qc = QuantumCircuit(num)
    params = ParameterVector(prefix, length=num // 2 * 3)
    for src, snk, p in zip(sources, sinks, params):
        layer = pool_circuit(p)
        qc.append(layer, [src, snk])
    return qc

# --------------------------------------------------------------------
# 2. Autoencoder‑style entanglement block
# --------------------------------------------------------------------
def auto_encoder_circuit(num_latent, num_trash):
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Entangling ansatz
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    qc.compose(ansatz, range(num_latent + num_trash), inplace=True)

    qc.barrier()
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])
    return qc

# --------------------------------------------------------------------
# 3. Hybrid QNN
# --------------------------------------------------------------------
def HybridQNN():
    algorithm_globals.random_seed = 42
    estimator = StatevectorEstimator()
    sampler = StatevectorSampler()

    # Feature map and QCNN ansatz
    feature_map = ZFeatureMap(8)
    ansatz = QuantumCircuit(8)

    # First convolution‑pool block
    ansatz.compose(conv_layer(8, "c1"), range(8), inplace=True)
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), range(8), inplace=True)

    # Second convolution‑pool block
    ansatz.compose(conv_layer(4, "c2"), range(4, 8), inplace=True)
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), range(4, 8), inplace=True)

    # Third convolution‑pool block
    ansatz.compose(conv_layer(2, "c3"), range(6, 8), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), range(6, 8), inplace=True)

    # Append autoencoder entanglement
    ae = auto_encoder_circuit(3, 2)

    # Assemble full circuit
    full_circuit = QuantumCircuit(8)
    full_circuit.compose(feature_map, range(8), inplace=True)
    full_circuit.compose(ansatz, range(8), inplace=True)
    full_circuit.compose(ae, range(8), inplace=True)

    observable = SparsePauliOp.from_list([("Z" * 8, 1)])

    qnn = EstimatorQNN(
        circuit=full_circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters + ae.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = ["HybridQNN"]
