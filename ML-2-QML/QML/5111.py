"""Hybrid quantum regression model built on Qiskit EstimatorQNN.

The circuit uses a parameterised Ry encoding, a random layer, and
RX/RZ rotations that act as trainable weights.  The observable is a
Pauli‑Z operator on all qubits.  The class inherits from
`qiskit_machine_learning.neural_networks.EstimatorQNN` so that it
behaves like a PyTorch module while still leveraging a quantum
backend.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RandomLayer
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
import numpy as np
from typing import List

class HybridQuantumRegression(QiskitEstimatorQNN):
    """
    A Qiskit EstimatorQNN that implements a hybrid quantum regression circuit.
    """

    def __init__(self, num_qubits: int = 4, num_layers: int = 3, input_dim: int = 2):
        # Parameterised input and weight gates
        input_params: List[Parameter] = [Parameter(f"x{i}") for i in range(input_dim)]
        weight_params: List[Parameter] = [Parameter(f"w{i}") for i in range(num_qubits)]

        # Build the variational circuit
        qc = QuantumCircuit(num_qubits)

        # Encoding: Ry rotations for each input feature (wrapped to the first `input_dim`
        # qubits; remaining qubits are unused in the encoding phase)
        for i, p in enumerate(input_params):
            qc.ry(p, i % num_qubits)

        # Random unitary layer to introduce entanglement
        qc.append(RandomLayer(num_qubits, n_ops=10), range(num_qubits))

        # Trainable single‑qubit rotations (RX + RZ) on every qubit
        for i in range(num_qubits):
            qc.rx(weight_params[i], i)
            qc.rz(weight_params[i], i)

        # Observable: Pauli‑Z on all qubits
        obs = SparsePauliOp.from_list([("Z" * num_qubits, 1.0)])

        super().__init__(
            circuit=qc,
            observables=obs,
            input_params=input_params,
            weight_params=weight_params,
            estimator=StatevectorEstimator(),
        )

    def __repr__(self) -> str:
        return f"<HybridQuantumRegression num_qubits={self.circuit.num_qubits} input_dim={len(self.input_params)}>"

__all__ = ["HybridQuantumRegression"]
