"""Hybrid quantum estimator that merges a variational regression circuit with a
quantum convolutional patch encoder.  It builds on the `EstimatorQNN` example
and the `Quanvolution` idea, producing a single EstimatorQNN instance that
can be used with Qiskit Machine Learning primitives.

The circuit operates on 4 qubits, each qubit representing a 2×2 pixel patch.
Input parameters encode the patch values via Ry rotations, a RandomLayer
entangles the qubits, and weight parameters are applied via Rz rotations.
Four Pauli‑Z observables give a 4‑dimensional feature vector that can be
used as the output of the quantum neural network.
"""

from __future__ import annotations

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator

def HybridEstimatorQuantum() -> EstimatorQNN:
    """Return a Qiskit EstimatorQNN that encodes a 2×2 patch on 4 qubits."""
    # Parameters: 4 inputs + 4 weights
    input_params = [Parameter(f"inp{i}") for i in range(4)]
    weight_params = [Parameter(f"w{i}") for i in range(4)]

    qc = QuantumCircuit(4)
    # Encode inputs
    for i, p in enumerate(input_params):
        qc.ry(p, i)

    # Random entangling layer: 8 random two‑qubit gates
    # For reproducibility we use a fixed sequence of CX and H gates
    for i in range(4):
        qc.h(i)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.cx(3, 0)

    # Weight rotations
    for i, p in enumerate(weight_params):
        qc.rz(p, i)

    # Observables: Pauli‑Z on each qubit
    observables = [
        SparsePauliOp.from_list([("Z" + "I" * (3 - i), 1)]) for i in range(4)
    ]

    estimator = StatevectorEstimator()
    estimator_qnn = EstimatorQNN(
        circuit=qc,
        observables=observables,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )
    return estimator_qnn

__all__ = ["HybridEstimatorQuantum"]
