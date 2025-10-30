"""
Hybrid quantum‑neural estimator that unites a quantum convolution layer
with a parameterized quantum linear layer.

The circuit is built from scratch using Qiskit primitives and is fed to
`qiskit_machine_learning.neural_networks.EstimatorQNN`.  The input
parameters model a quantum convolution over two qubits, while the
weight parameters correspond to a single‑qubit rotation that acts as a
quantum linear transformation.  The expectation value of `Z` on the first
qubit is used as the observable.
"""

from __future__ import annotations

from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator


def EstimatorQNN() -> QiskitEstimatorQNN:
    """Return an instance of the hybrid quantum estimator."""
    # Define input (convolution) parameters
    input_params = [Parameter(f"x{i}") for i in range(2)]
    # Define weight (quantum linear) parameters
    weight_params = [Parameter(f"w{i}") for i in range(2)]

    # Build the quantum circuit
    qc = QuantumCircuit(2)
    # Convolution-like layer: independent rotations + entanglement
    qc.rx(input_params[0], 0)
    qc.rx(input_params[1], 1)
    qc.cx(0, 1)
    # Quantum linear layer: rotations on each qubit
    qc.ry(weight_params[0], 0)
    qc.ry(weight_params[1], 1)

    # Observable: Pauli Z on first qubit
    observable = SparsePauliOp.from_list([("Z", 1)])

    estimator = Estimator()
    estimator_qnn = QiskitEstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )
    return estimator_qnn


__all__ = ["EstimatorQNN"]
