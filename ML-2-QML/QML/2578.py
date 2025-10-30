"""Quantum implementation of EstimatorQNN that encodes classical features into a 4‑qubit variational circuit.

The circuit accepts 4 input parameters (one per 2×2 image patch feature) and 4 trainable weight parameters.
It applies Ry rotations for the inputs, Rx rotations for the weights, entangles the qubits with CX gates,
and measures the Pauli‑Z expectation values.  The resulting expectation vector is linearly combined
by the EstimatorQNN wrapper to produce a scalar output.
"""

from __future__ import annotations

from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator

def EstimatorQNN():
    # Define 4 input parameters (one per feature) and 4 weight parameters
    input_params = [Parameter(f"inp_{i}") for i in range(4)]
    weight_params = [Parameter(f"w_{i}") for i in range(4)]

    # Build a 4‑qubit circuit
    qc = QuantumCircuit(4)
    # Encode inputs
    for i, p in enumerate(input_params):
        qc.ry(p, i)
    # Encode weights
    for i, p in enumerate(weight_params):
        qc.rx(p, i)
    # Entangle qubits
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)

    # Measurement: Pauli‑Z expectation values for each qubit
    pauli_strings = []
    for i in range(4):
        pauli = ["I"] * 4
        pauli[i] = "Z"
        pauli_strings.append("".join(pauli))
    observables = [SparsePauliOp.from_list([(s, 1)]) for s in pauli_strings]

    # Wrap with EstimatorQNN
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
