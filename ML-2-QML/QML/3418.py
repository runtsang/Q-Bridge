"""Quantum estimator that mirrors the classical HybridEstimatorQNN structure."""
from __future__ import annotations

from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator

def HybridEstimatorQNN(num_wires: int, num_weight_params: int):
    """
    Build a quantum neural network that accepts classical features encoded as rotation angles
    and learns variational weight parameters.

    Parameters
    ----------
    num_wires : int
        Number of qubits in the circuit.
    num_weight_params : int
        Number of trainable variational parameters.

    Returns
    -------
    EstimatorQNN
        A Qiskit Machine Learning estimator ready for training.
    """
    # Parameter vectors
    input_params = ParameterVector("x", 2 * num_wires)
    weight_params = ParameterVector("w", num_weight_params)

    qc = QuantumCircuit(num_wires)

    # --- Input encoding (RX, RY for each qubit) ---
    for q in range(num_wires):
        qc.rx(input_params[2 * q], q)
        qc.ry(input_params[2 * q + 1], q)

    # --- Entangling layer (simple chain of CX gates) ---
    for q in range(num_wires - 1):
        qc.cx(q, q + 1)

    # --- Variational layer (RZ rotations) ---
    for q in range(num_wires):
        idx = q % num_weight_params
        qc.rz(weight_params[idx], q)

    # Measurement observable: product of Z on all qubits
    observable = SparsePauliOp.from_list([("Z" * num_wires, 1)])

    # Estimator backend
    estimator = StatevectorEstimator()

    return EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )

__all__ = ["HybridEstimatorQNN"]
