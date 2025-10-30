"""Quantum module for FraudDetectionHybrid model.

Provides a Qiskit circuit that encodes input features, applies a QCNN‑like convolution and pooling structure,
followed by a quantum fully connected layer, and returns expectation values of PauliZ on each qubit.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

def build_quantum_circuit(n_qubits: int = 4) -> QuantumCircuit:
    """Build the full quantum circuit used in the hybrid model."""
    qc = QuantumCircuit(n_qubits)
    # Photonic‑inspired feature map
    fm_params = ParameterVector("fm", length=n_qubits)
    for i in range(n_qubits):
        qc.ry(fm_params[i], i)
        qc.rz(fm_params[i] + np.pi / 4, i)
    # QCNN convolutional layer
    conv_params = ParameterVector("conv", length=n_qubits)
    for i in range(0, n_qubits - 1, 2):
        qc.cx(i, i + 1)
        qc.rx(conv_params[i], i)
        qc.ry(conv_params[i + 1], i + 1)
    # QCNN pooling layer
    pool_params = ParameterVector("pool", length=n_qubits)
    for i in range(0, n_qubits - 1, 2):
        qc.cx(i, i + 1)
        qc.rz(pool_params[i], i)
        qc.ry(pool_params[i + 1], i + 1)
    # Quantum fully connected layer
    qfc_params = ParameterVector("qfc", length=n_qubits)
    for i in range(n_qubits):
        qc.rx(qfc_params[i], i)
        qc.ry(qfc_params[i], i)
    # Measurement
    qc.measure_all()
    return qc

__all__ = ["build_quantum_circuit"]
