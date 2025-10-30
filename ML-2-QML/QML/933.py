"""Variational quantum neural network with two qubits and entanglement.

The circuit consists of:
  * H on both qubits.
  * Parameterised RY/RZ rotations on each qubit (input parameters).
  * Entangling CNOT between qubits.
  * Additional parameterised rotations (weight parameters).
  * Observables: Pauli Y on each qubit.
  * Uses Qiskit Machine Learning EstimatorQNN with a StatevectorEstimator.

This module demonstrates how to build a richer quantum circuit for regression tasks.
"""

from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator

def EstimatorQNN() -> QiskitEstimatorQNN:
    """Return a 2‑qubit variational quantum neural network."""
    # Parameter vectors
    input_params = ParameterVector("input", 2)   # one per qubit
    weight_params = ParameterVector("weight", 4) # two rotations per qubit

    qc = QuantumCircuit(2)

    # Initial Hadamards
    qc.h(0)
    qc.h(1)

    # Input rotations
    qc.ry(input_params[0], 0)
    qc.rz(input_params[1], 1)

    # Entanglement
    qc.cx(0, 1)

    # Weight rotations
    qc.rx(weight_params[0], 0)
    qc.rz(weight_params[1], 0)
    qc.rx(weight_params[2], 1)
    qc.rz(weight_params[3], 1)

    # Observables: Pauli Y on each qubit
    observables = SparsePauliOp.from_list([("YI", 1), ("IY", 1)])

    estimator = StatevectorEstimator()
    estimator_qnn = QiskitEstimatorQNN(
        circuit=qc,
        observables=observables,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )
    return estimator_qnn

__all__ = ["EstimatorQNN"]
