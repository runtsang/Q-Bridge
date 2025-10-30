"""quantum_encoder_circuit.py"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

def get_quantum_encoder(num_qubits: int, input_dim: int) -> EstimatorQNN:
    """
    Construct a quantum neural network that maps an input vector of length `input_dim`
    to a latent vector of length `num_qubits`. The circuit uses a RealAmplitudes
    ansatz on the ancilla qubits and encodes the input data as Ry rotations.
    """
    # Parameters for encoding input data
    data_params = [Parameter(f"d_{i}") for i in range(input_dim)]

    # Build circuit
    qc = QuantumCircuit(input_dim + num_qubits, name="quantum_encoder")

    # Encode input data as Ry rotations on first `input_dim` qubits
    for i, p in enumerate(data_params):
        qc.ry(p, i)

    # Apply ansatz on ancilla qubits
    ancilla_start = input_dim
    ansatz = RealAmplitudes(num_qubits, reps=2)
    qc.append(ansatz, range(ancilla_start, ancilla_start + num_qubits))

    # Define observables: Pauli Z on each ancilla qubit
    observables = []
    for i in range(num_qubits):
        pauli_str = "I" * input_dim + "Z" * i + "I" * (num_qubits - i - 1)
        observables.append(SparsePauliOp.from_list([(pauli_str, 1)]))

    estimator = StatevectorEstimator()
    qnn = EstimatorQNN(
        circuit=qc,
        observables=observables,
        input_params=data_params,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn
