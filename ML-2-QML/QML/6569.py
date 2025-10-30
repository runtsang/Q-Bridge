"""Quantum neural network with entanglement and multi‑qubit observables."""

from __future__ import annotations

from typing import Sequence
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator


def _build_circuit(num_qubits: int, input_params: Sequence[Parameter], weight_params: Sequence[Parameter]) -> QuantumCircuit:
    """Constructs a layered circuit with entanglement."""
    qc = QuantumCircuit(num_qubits)
    # Input encoding: rotate each qubit about Y with input param
    for i in range(num_qubits):
        qc.ry(input_params[i], i)
    # First entangling layer
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    # Parameterised rotation layer
    for i in range(num_qubits):
        qc.rx(weight_params[i], i)
        qc.rz(weight_params[i], i)
    # Second entangling layer
    for i in range(num_qubits - 1):
        qc.cx(i + 1, i)
    return qc


def EstimatorQNN(num_qubits: int = 3) -> QiskitEstimatorQNN:
    """
    Returns a qiskit EstimatorQNN instance built from a custom circuit.

    Parameters
    ----------
    num_qubits : int, default 3
        Number of qubits in the variational circuit.

    Returns
    -------
    QiskitEstimatorQNN
        A qiskit neural network estimator ready to be trained.
    """
    # Parameters
    input_params = [Parameter(f"x{i}") for i in range(num_qubits)]
    weight_params = [Parameter(f"w{i}") for i in range(num_qubits)]
    # Build circuit
    qc = _build_circuit(num_qubits, input_params, weight_params)

    # Observables: use multi‑qubit Pauli Y tensor product
    observable = SparsePauliOp.from_list([("Y" * num_qubits, 1.0)])

    # Estimator primitive
    estimator = StatevectorEstimator()

    # Construct EstimatorQNN
    return QiskitEstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )


__all__ = ["EstimatorQNN"]
