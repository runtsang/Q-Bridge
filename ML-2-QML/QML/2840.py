"""Quantum variational encoder for the hybrid autoencoder.

This module builds a quantum neural network that maps a classical feature
vector (output of the QCNN) into a latent space.  It uses a Z‑feature
map to encode the classical data followed by a RealAmplitudes ansatz.
The latent vector is obtained from the expectation values of Z on each
latent qubit.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp


def QuantumConvolutionAutoencoderQNN(
    input_dim: int = 8,
    latent_dim: int = 4,
) -> EstimatorQNN:
    """Build a quantum neural network for the hybrid autoencoder.

    Parameters
    ----------
    input_dim : int, default 8
        Dimensionality of the classical feature vector from the QCNN.
    latent_dim : int, default 4
        Number of qubits in the latent space (also the size of the latent vector).

    Returns
    -------
    EstimatorQNN
        QNN that can be used for forward passes and parameter updates.
    """
    algorithm_globals.random_seed = 42
    estimator = Estimator()

    # Feature map that encodes the classical data
    feature_map = ZFeatureMap(input_dim)

    # Variational ansatz that produces the latent representation
    ansatz = RealAmplitudes(latent_dim, reps=2)

    # Build the combined circuit
    circuit = QuantumCircuit(input_dim + latent_dim)
    # Apply feature map to the first ``input_dim`` qubits
    circuit.compose(feature_map, range(input_dim), inplace=True)
    # Apply ansatz to the remaining ``latent_dim`` qubits
    circuit.compose(ansatz, range(input_dim, input_dim + latent_dim), inplace=True)

    # Observables: Pauli‑Z on each latent qubit
    observables = []
    for i in range(latent_dim):
        pauli_str = "I" * i + "Z" + "I" * (latent_dim - i - 1)
        observables.append(SparsePauliOp.from_list([(pauli_str, 1)]))

    qnn = EstimatorQNN(
        circuit=circuit,
        observables=observables,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn


def quantum_encode(qnn: EstimatorQNN, inputs: np.ndarray) -> np.ndarray:
    """Encode a batch of classical inputs using the quantum circuit.

    Parameters
    ----------
    qnn : EstimatorQNN
        The quantum neural network.
    inputs : np.ndarray
        2‑D array of shape ``(batch_size, input_dim)``.

    Returns
    -------
    np.ndarray
        2‑D array of shape ``(batch_size, latent_dim)`` containing the
        expectation values of the latent qubits.
    """
    return qnn.predict(inputs)


__all__ = ["QuantumConvolutionAutoencoderQNN", "quantum_encode"]
