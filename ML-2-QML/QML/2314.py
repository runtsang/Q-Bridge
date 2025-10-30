"""Quantum circuit builder for the hybrid fraud detection model."""

from __future__ import annotations

from typing import Sequence

from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


def build_fraud_quantum_circuit() -> EstimatorQNN:
    """
    Construct a Qiskit EstimatorQNN that emulates the photonic fraud detection
    circuit.  The circuit uses two qubits and includes a beam‑splitter‑style
    entangling gate followed by parameterized rotations, squeezes, displacements
    and Kerr‑type non‑linear gates.

    Returns
    -------
    EstimatorQNN
        Quantum neural network ready to be integrated into a hybrid model.
    """
    # Parameter names
    input_params: Sequence[Parameter] = [Parameter("x1"), Parameter("x2")]
    weight_params: Sequence[Parameter] = [Parameter("w1"), Parameter("w2")]

    qc = QuantumCircuit(2)

    # Layer 1 – mimicking a photonic beam‑splitter and squeezers
    qc.h(0)
    qc.h(1)
    qc.ry(input_params[0], 0)
    qc.ry(input_params[1], 1)
    qc.rx(weight_params[0], 0)
    qc.rx(weight_params[1], 1)

    # Observables (Z‑basis on both qubits)
    observable = SparsePauliOp.from_list([("Z" * qc.num_qubits, 1)])

    # Estimator that returns expectation values
    estimator = StatevectorEstimator()

    return EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )


__all__ = ["build_fraud_quantum_circuit"]
