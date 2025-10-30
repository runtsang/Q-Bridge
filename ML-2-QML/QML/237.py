"""
Deep, entangled parameterised quantum neural network for regression.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator as Estimator


def EstimatorQNN(
    num_qubits: int = 2,
    layers: int = 3,
    input_dim: int = 2,
    backend: str = "statevector_simulator",
) -> QiskitEstimatorQNN:
    """
    Build a parameterised quantum circuit with multiple entangling layers.

    Parameters
    ----------
    num_qubits : int
        Number of qubits to use in the ansatz.
    layers : int
        Number of repeatable layers in the circuit.
    input_dim : int
        Number of input parameters (must not exceed num_qubits).
    backend : str
        Backend name for the StatevectorEstimator.

    Returns
    -------
    qiskit_machine_learning.neural_networks.EstimatorQNN
        Configured quantum neural network ready for training.
    """
    if input_dim > num_qubits:
        raise ValueError("input_dim cannot exceed num_qubits")

    # Input encoding parameters
    input_params = [Parameter(f"input_{i}") for i in range(input_dim)]

    # Weight parameters for each qubit in each layer
    weight_params = [
        Parameter(f"theta_{l}_{q}") for l in range(layers) for q in range(num_qubits)
    ]

    qc = QuantumCircuit(num_qubits)

    # Input encoding
    for q in range(num_qubits):
        if q < input_dim:
            qc.ry(input_params[q], q)

    # Parameterised layers with entanglement
    for l in range(layers):
        for q in range(num_qubits):
            idx = l * num_qubits + q
            qc.rx(weight_params[idx], q)
        # Entangling CZ between neighbouring qubits
        for q in range(num_qubits - 1):
            qc.cz(q, q + 1)

    # Observable: sum of Z on each qubit (single‑observable regression)
    observable = SparsePauliOp.from_list([("Z" * num_qubits, 1)])

    estimator = Estimator(backend=backend)

    return QiskitEstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )


__all__ = ["EstimatorQNN"]
