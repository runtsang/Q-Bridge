"""Quantum neural network with a two‑qubit variational circuit.

The original EstimatorQNN used a single‑qubit circuit with a single observable.
This version expands the circuit to two qubits, introduces multiple layers of
parameterised rotations and entangling gates, and measures two Pauli observables.
It remains a drop‑in replacement for the original API but provides richer expressivity.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as _EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator


def EstimatorQNN() -> _EstimatorQNN:
    """
    Build a 2‑qubit variational circuit with 2 rotation‑entanglement layers.

    Returns
    -------
    EstimatorQNN
        A Qiskit EstimatorQNN instance ready for training.
    """
    # Parameters for input data (one per qubit)
    input_params = [Parameter(f"inp{i}") for i in range(2)]

    # Weight parameters for each rotation in each layer
    weight_params = [
        Parameter(f"w{layer}{gate}{qubit}")
        for layer in range(2)
        for gate in ("x", "y", "z")
        for qubit in range(2)
    ]

    # Build the circuit
    qc = QuantumCircuit(2)
    # First layer
    for q in range(2):
        qc.ry(input_params[q], q)  # encode data
        qc.rx(weight_params[0 * 6 + q * 3 + 0], q)  # x rotation
        qc.ry(weight_params[0 * 6 + q * 3 + 1], q)  # y rotation
        qc.rz(weight_params[0 * 6 + q * 3 + 2], q)  # z rotation
    # Entangling
    qc.cx(0, 1)
    qc.cx(1, 0)
    # Second layer
    for q in range(2):
        qc.rx(weight_params[1 * 6 + q * 3 + 0], q)
        qc.ry(weight_params[1 * 6 + q * 3 + 1], q)
        qc.rz(weight_params[1 * 6 + q * 3 + 2], q)
    # Entangling
    qc.cx(0, 1)
    qc.cx(1, 0)
    qc.barrier()

    # Observables: Y⊗Y and Z⊗Z
    observables = SparsePauliOp.from_list(
        [("YY", 1.0), ("ZZ", 1.0)]
    )

    # Create the EstimatorQNN
    estimator = StatevectorEstimator()
    estimator_qnn = _EstimatorQNN(
        circuit=qc,
        observables=observables,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )

    return estimator_qnn


__all__ = ["EstimatorQNN"]
