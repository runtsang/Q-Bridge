"""Quantum QCNN–Autoencoder hybrid circuit.

The quantum circuit builds a QCNN ansatz (convolution + pooling layers) and
augments it with an autoencoder sub‑circuit that uses a swap‑test to
compress the state into a latent register.  The resulting EstimatorQNN
can be trained end‑to‑end with a classical optimiser.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals


def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution unitary used in QCNN layers."""
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


def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling unitary."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def _conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Apply convolution blocks pairwise across the register."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits // 2 * 3)
    for i in range(0, num_qubits, 2):
        sub = _conv_circuit(params[i // 2 * 3 : i // 2 * 3 + 3])
        qc.append(sub, [i, i + 1])
    return qc


def _pool_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Apply pooling blocks pairwise across the register."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits // 2 * 3)
    for i in range(0, num_qubits, 2):
        sub = _pool_circuit(params[i // 2 * 3 : i // 2 * 3 + 3])
        qc.append(sub, [i, i + 1])
    return qc


def _autoencoder_subcircuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """
    Implements a swap‑test based autoencoder that compresses the state into
    ``num_latent`` qubits.  The remaining ``num_trash`` qubits are discarded.
    """
    total = num_latent + 2 * num_trash + 1
    qc = QuantumCircuit(total)
    # Encode part
    qc.append(RealAmplitudes(num_latent + num_trash, reps=3), list(range(num_latent + num_trash)))
    qc.barrier()
    # Swap‑test
    aux = total - 1
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, 0)  # measurement is ignored by the estimator
    return qc


def QCNNAutoencoder(num_qubits: int = 8, latent_dim: int = 3, trash_dim: int = 2) -> EstimatorQNN:
    """
    Build a hybrid QCNN–Autoencoder circuit and wrap it in an EstimatorQNN.

    Parameters
    ----------
    num_qubits : int
        Size of the input register (must be a power of two for the QCNN pattern).
    latent_dim : int
        Number of qubits that hold the compressed latent state.
    trash_dim : int
        Number of auxiliary qubits used for the autoencoder swap‑test.
    """
    algorithm_globals.random_seed = 42
    estimator = StatevectorEstimator()

    # Feature map
    feature_map = ZFeatureMap(num_qubits)
    # QCNN ansatz
    ansatz = QuantumCircuit(num_qubits)
    ansatz.compose(_conv_layer(num_qubits, "c1"), inplace=True)
    ansatz.compose(_pool_layer(num_qubits, "p1"), inplace=True)
    ansatz.compose(_conv_layer(num_qubits // 2, "c2"), inplace=True)
    ansatz.compose(_pool_layer(num_qubits // 2, "p2"), inplace=True)
    ansatz.compose(_conv_layer(num_qubits // 4, "c3"), inplace=True)
    ansatz.compose(_pool_layer(num_qubits // 4, "p3"), inplace=True)

    # Autoencoder sub‑circuit
    ae_circ = _autoencoder_subcircuit(latent_dim, trash_dim)

    # Full circuit
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)
    circuit.compose(ae_circ, inplace=True)

    # Observable to read out the latent value
    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

    return EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters + ae_circ.parameters,
        estimator=estimator,
    )
