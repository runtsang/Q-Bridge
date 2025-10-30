"""Quantum estimator that uses a parameterized ansatz and the Qiskit EstimatorQNN API."""

from __future__ import annotations

from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

def EstimatorQNNGen338(num_qubits: int = 2) -> EstimatorQNN:
    """
    Construct a variational quantum circuit for regression.
    Input parameters are encoded as Ry rotations on each qubit; weight parameters
    control a two‑layer entangled ansatz.  The observable is the sum of Pauli‑Z
    on all qubits.  The function returns a qiskit_machine_learning EstimatorQNN
    object that can be trained with a Qiskit backend.
    """
    # Input and weight parameters
    input_params = [Parameter(f"input_{i}") for i in range(num_qubits)]
    weight_params = [Parameter(f"weight_{i}") for i in range(num_qubits)]

    # Build the ansatz circuit
    qc = QuantumCircuit(num_qubits)
    # Input encoding
    for i in range(num_qubits):
        qc.ry(input_params[i], i)
    # First entangled layer
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    # Parameterized rotation layer
    for i in range(num_qubits):
        qc.rx(weight_params[i], i)
        qc.rz(weight_params[(i + 1) % num_qubits], i)
    # Second entangled layer
    for i in range(num_qubits - 1):
        qc.cx(i + 1, i)
    # Observable: sum of Z on all qubits
    observable = SparsePauliOp.from_list([("Z" * num_qubits, 1.0)])

    estimator = StatevectorEstimator()
    return EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )

__all__ = ["EstimatorQNNGen338"]
