"""Quantum estimator with a two‑qubit entangled variational circuit."""
from __future__ import annotations

from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as _EstimatorQNN
from qiskit.primitives import StatevectorEstimator

def EstimatorQNN() -> _EstimatorQNN:
    """
    Build and return a Qiskit EstimatorQNN with:
    * 2 qubits and a trainable entangling layer.
    * Parameterised Ry/Rz rotations per qubit.
    * Observable YY to capture correlations.
    """
    # Input parameters (two classical inputs)
    input_params = [Parameter("x1"), Parameter("x2")]

    # Weight parameters for the variational circuit
    weight_params = [Parameter(f"w{i}") for i in range(6)]

    # Construct the circuit
    qc = QuantumCircuit(2)

    # Encode classical inputs
    qc.ry(input_params[0], 0)
    qc.ry(input_params[1], 1)

    # Entanglement layer
    qc.cx(0, 1)

    # Parameterised rotation layers
    for i, qubit in enumerate([0, 1]):
        qc.ry(weight_params[2 * i], qubit)
        qc.rz(weight_params[2 * i + 1], qubit)

    # Optional second entanglement
    qc.cx(1, 0)

    # Observable: YY
    observable = SparsePauliOp.from_list([("YY", 1.0)])

    estimator = StatevectorEstimator()
    estimator_qnn = _EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )
    return estimator_qnn

__all__ = ["EstimatorQNN"]
