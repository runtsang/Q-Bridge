"""Quantum encoder that uses a QCNN ansatz and EstimatorQNN to transform latent representations."""

from __future__ import annotations

import numpy as np
from qiskit import Aer
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as EstimatorPrimitives


def _build_qcnn(num_qubits: int) -> QuantumCircuit:
    """
    Build a simple two‑layer QCNN ansatz as in the QCNN reference.
    Each layer contains a convolution and a pooling block applied to
    adjacent qubit pairs.
    """
    qc = QuantumCircuit(num_qubits)
    # Convolution parameters
    conv_params = ParameterVector("conv", length=3 * (num_qubits // 2))
    # Pooling parameters
    pool_params = ParameterVector("pool", length=3 * (num_qubits // 2))
    param_idx = 0
    for i in range(0, num_qubits, 2):
        # Convolution block
        qc.rz(-np.pi / 2, i)
        qc.cx(i, i + 1)
        qc.rz(conv_params[param_idx], i)
        qc.ry(conv_params[param_idx + 1], i + 1)
        qc.cx(i + 1, i)
        qc.ry(conv_params[param_idx + 2], i + 1)
        qc.cx(i, i + 1)
        qc.rz(np.pi / 2, i)
        param_idx += 3
        qc.barrier()

        # Pooling block
        qc.rz(-np.pi / 2, i)
        qc.cx(i, i + 1)
        qc.rz(pool_params[param_idx], i)
        qc.ry(pool_params[param_idx + 1], i + 1)
        qc.cx(i + 1, i)
        qc.ry(pool_params[param_idx + 2], i + 1)
        param_idx += 3
        qc.barrier()
    return qc


def _pauli_z_on_qubit(i: int, n: int) -> str:
    """Return a Pauli string with Z on qubit i and I elsewhere."""
    return "I" * i + "Z" + "I" * (n - i - 1)


def get_quantum_encoder(num_qubits: int, latent_dim: int) -> callable:
    """
    Return a function that maps a classical latent vector (batch, latent_dim)
    to a quantum latent vector using a QCNN ansatz and EstimatorQNN readout.
    """
    # Feature map to embed classical data into qubit states
    feature_map = ZFeatureMap(num_qubits)

    # QCNN ansatz
    ansatz = _build_qcnn(num_qubits)

    # Combined circuit
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    # Observables: Pauli‑Z on each qubit
    observables = [
        SparsePauliOp.from_list([( _pauli_z_on_qubit(i, num_qubits), 1 )])
        for i in range(num_qubits)
    ]

    # Estimator for expectation values
    estimator = EstimatorPrimitives()
    qnn = EstimatorQNN(
        circuit=circuit,
        observables=observables,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )

    def quantum_encoder(inputs: np.ndarray) -> np.ndarray:
        """
        Map inputs (batch, latent_dim) to quantum latent representation.
        The input dimension must match latent_dim and num_qubits.
        """
        if inputs.shape[1]!= latent_dim:
            raise ValueError(f"Input dimension {inputs.shape[1]} must equal latent_dim {latent_dim}.")
        return qnn.forward(inputs)

    return quantum_encoder


__all__ = ["get_quantum_encoder"]
