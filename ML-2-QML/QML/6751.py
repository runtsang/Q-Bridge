"""Quantum circuit used by :class:`HybridEstimatorQNN`."""

from __future__ import annotations

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes

def hybrid_quantum_circuit(latent_dim: int, reps: int = 3) -> QuantumCircuit:
    """
    Builds a variational circuit that takes ``latent_dim`` input parameters
    followed by a set of trainable weight parameters.

    Parameters
    ----------
    latent_dim : int
        Number of qubits that carry the encoded classical data.
    reps : int, optional
        Number of repetitions of the RealAmplitudes ansatz. Default is 3.

    Returns
    -------
    QuantumCircuit
        The constructed circuit with parameters:
        * first ``latent_dim`` parameters are input parameters.
        * remaining parameters belong to the ansatz and are trainable.
    """
    # Define parameter names
    input_params = [Parameter(f"x{i}") for i in range(latent_dim)]

    # Remaining parameters for the ansatz
    # RealAmplitudes automatically creates a Parameter list of size
    # 2 * num_qubits * reps + num_qubits
    dummy_circ = RealAmplitudes(latent_dim, reps=reps)
    weight_params = dummy_circ.parameters

    qc = QuantumCircuit(latent_dim)
    # Encode classical data via RX rotations
    for idx, param in enumerate(input_params):
        qc.rx(param, idx)

    # Apply the ansatz
    qc.compose(dummy_circ, inplace=True)

    # Measure expectation of Z on the first qubit
    qc.measure_all()

    # Attach parameters
    qc.params = input_params + weight_params

    return qc
