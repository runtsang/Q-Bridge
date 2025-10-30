"""Quantum hybrid autoencoder built from QCNN‑style convolutional layers and a swap‑test decoder.

The function `HybridAutoencoder` constructs a variational circuit that
encodes classical data into a latent subspace using a QCNN‑style variational ansatz
and decodes it with a simple ancilla measurement.  The returned `SamplerQNN` can
be used as a differentiable layer in a hybrid training loop; the quantum part
is treated as a fixed feature extractor.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals


def _conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """QCNN‑inspired convolutional layer with two‑qubit interactions."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        qc.cx(i, i + 1)
        qc.rz(params[i // 2 * 3], i)
        qc.ry(params[i // 2 * 3 + 1], i + 1)
        qc.cx(i, i + 1)
    return qc


def _pool_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """QCNN‑inspired pooling layer that reduces entanglement."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        qc.cx(i, i + 1)
        qc.rz(params[i // 2 * 3], i)
        qc.ry(params[i // 2 * 3 + 1], i + 1)
        qc.cx(i, i + 1)
    return qc


def HybridAutoencoder(
    input_dim: int,
    latent_dim: int,
    *,
    n_layers: int = 3,
    seed: int = 42,
) -> SamplerQNN:
    """
    Quantum hybrid autoencoder that encodes classical data into a latent subspace
    using a QCNN‑style variational ansatz and decodes it with a single ancilla
    measurement.  The returned `SamplerQNN` can be integrated into a hybrid
    training pipeline.
    """
    algorithm_globals.random_seed = seed
    sampler = StatevectorSampler()

    # Feature map to embed classical data
    feature_map = ZFeatureMap(input_dim)

    # Encoder ansatz (QCNN style)
    encoder = QuantumCircuit(latent_dim + input_dim)
    for layer in range(n_layers):
        encoder.compose(
            _conv_layer(latent_dim, f"enc{layer}_c"), range(latent_dim), inplace=True
        )
        encoder.compose(
            _pool_layer(latent_dim, f"enc{layer}_p"), range(latent_dim), inplace=True
        )

    # Simple ancilla measurement decoder
    qr = QuantumRegister(latent_dim + 1, "q")
    cr = ClassicalRegister(1, "c")
    decoder = QuantumCircuit(qr, cr)
    decoder.h(qr[latent_dim])  # ancilla
    decoder.measure(qr[latent_dim], cr[0])

    # Assemble full circuit
    circuit = QuantumCircuit(input_dim + latent_dim + 1)
    circuit.compose(feature_map, range(input_dim), inplace=True)
    circuit.compose(encoder, range(input_dim, input_dim + latent_dim), inplace=True)
    circuit.compose(decoder, range(input_dim + latent_dim), inplace=True)

    # QNN
    qnn = SamplerQNN(
        circuit=circuit,
        input_params=feature_map.parameters,
        weight_params=encoder.parameters + decoder.parameters,
        interpret=lambda x: x[0],  # probability of measuring ancilla in |1>
        output_shape=1,
        sampler=sampler,
    )
    return qnn


__all__ = ["HybridAutoencoder"]
