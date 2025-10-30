"""Quantum implementation of EstimatorQNN using a variational circuit.

The circuit operates on two qubits, uses input‑dependent Ry rotations
followed by weight‑dependent Rz rotations, entangles the qubits with a
CNOT gate, and measures the Z⊗Z observable.  The class implements a
callable interface that accepts a list of two input values and returns
the expectation value, mirroring the behaviour of the classical
`EstimatorQNN`."""
from __future__ import annotations

from typing import List

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator


class EstimatorQNN:
    """
    Variational quantum circuit wrapped in the Qiskit EstimatorQNN class.
    Provides a callable that evaluates the circuit for given input angles.
    """

    def __init__(self) -> None:
        # Define parameters
        input_params = [Parameter(f"input_{i}") for i in range(2)]
        weight_params = [Parameter(f"weight_{i}") for i in range(4)]
        all_params = input_params + weight_params

        # Build the circuit
        qc = QuantumCircuit(2)
        # Input rotations
        qc.ry(input_params[0], 0)
        qc.ry(input_params[1], 1)
        # Weight rotations
        qc.rz(weight_params[0], 0)
        qc.rz(weight_params[1], 1)
        qc.rz(weight_params[2], 0)
        qc.rz(weight_params[3], 1)
        # Entanglement
        qc.cx(0, 1)

        # Observable for expectation value
        observable = SparsePauliOp.from_list([("ZZ", 1)])

        # Create the EstimatorQNN object
        self._qnn = QiskitEstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=input_params,
            weight_params=weight_params,
            estimator=Estimator(),
        )

    def __call__(self, inputs: List[float]) -> float:
        """
        Evaluate the circuit for the provided inputs.

        Args:
            inputs: List of two floats corresponding to the input
                    parameters.  The weight parameters are optimised
                    via the Qiskit Estimator during training.

        Returns:
            The expectation value of the observable as a float.
        """
        return float(self._qnn.predict(inputs))

__all__ = ["EstimatorQNN"]
