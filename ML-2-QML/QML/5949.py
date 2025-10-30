"""Quantum neural network with deeper variational circuit and tunable observable."""
from __future__ import annotations

from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as EstimatorQNNCls
from qiskit.primitives import StatevectorEstimator

def EstimatorQNN() -> EstimatorQNNCls:
    """
    Return a variational quantum circuit that accepts two classical inputs
    and estimates an expectation value of a tunable observable.
    The circuit uses two qubits, data re‑uploading, and a layered ansatz.
    """
    # Classical input parameters
    input_params = [Parameter("x1"), Parameter("x2")]

    # Weight parameters for the variational layers
    weight_params = [Parameter(f"w{i}") for i in range(8)]

    # Build the circuit
    qc = QuantumCircuit(2)

    # Data re‑uploading: encode inputs as Ry rotations
    qc.ry(input_params[0], 0)
    qc.ry(input_params[1], 1)

    # Layered ansatz (two layers)
    for i in range(2):
        # Single‑qubit rotations
        qc.ry(weight_params[2 * i], 0)
        qc.rz(weight_params[2 * i + 1], 0)
        qc.ry(weight_params[2 * i + 2], 1)
        qc.rz(weight_params[2 * i + 3], 1)
        # Entangling
        qc.cx(0, 1)
        qc.cx(1, 0)

    # Observable: weighted sum of Z on qubit 0 and X on qubit 1
    observable = SparsePauliOp.from_list(
        [("Z" * qc.num_qubits, 0.6), ("X" * qc.num_qubits, 0.4)]
    )

    # Instantiate the estimator
    estimator = StatevectorEstimator()

    # Wrap into Qiskit Machine Learning EstimatorQNN
    estimator_qnn = EstimatorQNNCls(
        circuit=qc,
        observables=observable,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )

    return estimator_qnn


__all__ = ["EstimatorQNN"]
