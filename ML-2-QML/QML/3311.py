"""Quantum neural network that encodes two inputs, entangles them, and applies
weight rotations.  The expectation value of Pauli‑Z on the first qubit
serves as the regression output.  This design blends the EstimatorQNN
variational circuit with a quantum convolutional motif."""

from __future__ import annotations

from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN as QuantumEstimatorQNN

def EstimatorQNN():
    """
    Return a quantum neural network that performs regression using a
    2‑qubit variational circuit.  Two inputs are encoded via Ry gates,
    the qubits are entangled with a CNOT, and weight rotations are applied.
    The expectation of Pauli‑Z on qubit 0 is used as the output.
    """
    # Define circuit parameters
    x1 = Parameter("x1")
    x2 = Parameter("x2")
    w1 = Parameter("w1")
    w2 = Parameter("w2")

    qc = QuantumCircuit(2)
    # Input encoding
    qc.ry(x1, 0)
    qc.ry(x2, 1)
    # Entanglement (quantum convolution motif)
    qc.cx(0, 1)
    # Weight rotations
    qc.rx(w1, 0)
    qc.rx(w2, 1)

    # Observable: Pauli‑Z on qubit 0
    observable = SparsePauliOp.from_list([("Z", 1)])

    estimator = StatevectorEstimator()

    qnn = QuantumEstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=[x1, x2],
        weight_params=[w1, w2],
        estimator=estimator,
    )
    return qnn

__all__ = ["EstimatorQNN"]
