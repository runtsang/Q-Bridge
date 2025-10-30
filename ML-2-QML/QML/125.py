"""Quantum neural network with a 2‑qubit variational circuit.

The circuit uses alternating layers of single‑qubit rotations and CNOT
entangling gates.  Two observables (Pauli‑Z on each qubit) are measured,
providing a richer feature vector for regression.  Parameters are split
into input and trainable weight sets, mirroring the classical model.

The function returns an instance of Qiskit’s `EstimatorQNN` ready for
hybrid training with a chosen backend or simulator.
"""

from __future__ import annotations

from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp


def EstimatorQNN() -> QiskitEstimatorQNN:
    """Build and return a 2‑qubit variational estimator network."""
    # Parameters
    input_params = [Parameter(f"x{i}") for i in range(2)]   # two input angles
    weight_params = [Parameter(f"w{i}") for i in range(4)]  # four trainable angles

    # Variational circuit
    qc = QuantumCircuit(2)
    # Input encoding: RX rotations on each qubit
    qc.rx(input_params[0], 0)
    qc.rx(input_params[1], 1)
    # Entangling layer
    qc.cx(0, 1)
    # Parameterised rotation layer
    qc.ry(weight_params[0], 0)
    qc.ry(weight_params[1], 1)
    qc.cx(1, 0)
    qc.ry(weight_params[2], 0)
    qc.ry(weight_params[3], 1)
    qc.cx(0, 1)

    # Observables: Pauli‑Z on each qubit
    observables = SparsePauliOp.from_list([("Z" * qc.num_qubits, 1), ("Z" * qc.num_qubits, 1)])

    # Estimator backend (statevector for exact gradients)
    estimator = StatevectorEstimator()

    # Construct the EstimatorQNN
    estimator_qnn = QiskitEstimatorQNN(
        circuit=qc,
        observables=observables,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )
    return estimator_qnn


__all__ = ["EstimatorQNN"]
