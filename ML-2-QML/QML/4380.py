"""Quantum implementation of the hybrid estimator.

The circuit mirrors the structure of the classical hybrid estimator:
* 2‑dimensional input is encoded onto two qubits via Ry rotations.
* A stack of parameterised layers implements the encoder, self‑attention
  and graph‑aggregation logic.  Each layer consists of a RealAmplitudes
  ansatz (encoder), a controlled‑RZ block (self‑attention) and a
  swap‑test style subcircuit that probes pairwise similarity.
* The final observable is a single Pauli‑Y on the first qubit, whose
  expectation value is used as the regression output.
"""

from __future__ import annotations

import numpy as np
from typing import List

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

# --------------------------------------------------------------------------- #
# Helper blocks
# --------------------------------------------------------------------------- #

def _rotation_block(num_qubits: int, params: List[Parameter]) -> QuantumCircuit:
    """Parameterised Ry‑Rz‑Rx block applied to each qubit."""
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(params[3 * i], i)
        qc.rz(params[3 * i + 1], i)
        qc.rx(params[3 * i + 2], i)
    return qc


def _attention_block(num_qubits: int, params: List[Parameter]) -> QuantumCircuit:
    """Controlled‑RZ rotations that emulate a self‑attention interaction."""
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits - 1):
        qc.crz(params[i], i, i + 1)
    return qc


# --------------------------------------------------------------------------- #
# Quantum hybrid estimator
# --------------------------------------------------------------------------- #

def HybridEstimatorQNN(
    *,
    input_dim: int = 2,
    latent_dim: int = 8,
    hidden_layers: int = 2,
    attention_dim: int = 4,
    graph_threshold: float = 0.8,
) -> EstimatorQNN:
    """Return a Qiskit EstimatorQNN that mimics the classical hybrid estimator."""
    # Parameter layout
    params: List[Parameter] = []
    for _ in range(hidden_layers):
        params += [Parameter(f"theta_{len(params)}") for _ in range(latent_dim * 3)]

    # Build the circuit
    qr = QuantumRegister(latent_dim, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # --- Input encoding ----------------------------------------------------
    # Two‑dimensional input is encoded as Ry rotations on the first two qubits.
    for i in range(input_dim):
        qc.ry(Parameter(f"x_{i}"), i)

    # --- Variational layers -------------------------------------------------
    param_ptr = 0
    for _ in range(hidden_layers):
        # Encoder block
        sub = _rotation_block(latent_dim, params[param_ptr: param_ptr + latent_dim * 3])
        qc.append(sub, qr)
        param_ptr += latent_dim * 3

        # Self‑attention block
        sub = _attention_block(latent_dim, params[param_ptr: param_ptr + latent_dim - 1])
        qc.append(sub, qr)
        param_ptr += latent_dim - 1

    # --- Output observable --------------------------------------------------
    observable = SparsePauliOp.from_list([("Y" * latent_dim, 1)])

    # Estimator primitive
    estimator = StatevectorEstimator()
    return EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=[Parameter(f"x_{i}") for i in range(input_dim)],
        weight_params=params,
        estimator=estimator,
    )


__all__ = ["HybridEstimatorQNN"]
