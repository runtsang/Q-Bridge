"""Variational quantum neural network with entanglement and multiple observables.

The circuit encodes 2‑dimensional data into a 2‑qubit ansatz and trains
via a StatevectorEstimator.  The design is deliberately modular so the
model can be swapped for other ansätze or simulators.
"""

from __future__ import annotations

from typing import List
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator

class EstimatorQNN:
    """
    Variational quantum neural network for regression.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    depth : int
        Number of repeated entangling layers.
    input_params : List[Parameter] | None
        Parameters used for data encoding.  If None, two parameters are
        created for the 2‑dimensional input.
    weight_params : List[Parameter] | None
        Trainable parameters of the ansatz.  If None, they are created
        automatically.
    """

    def __init__(self,
                 num_qubits: int = 2,
                 depth: int = 2,
                 input_params: List[Parameter] | None = None,
                 weight_params: List[Parameter] | None = None) -> None:
        self.num_qubits = num_qubits
        self.depth = depth

        # Build data‑encoding parameters
        if input_params is None:
            input_params = [Parameter(f"data_{i}") for i in range(num_qubits)]
        self.input_params = input_params

        # Build trainable parameters
        if weight_params is None:
            weight_params = [Parameter(f"w_{d}_{q}") for d in range(depth) for q in range(num_qubits)]
        self.weight_params = weight_params

        # Construct the variational circuit
        self.circuit = self._build_circuit()

        # Observable: sum of Z on all qubits
        self.observable = SparsePauliOp.from_list([("Z" * num_qubits, 1)])

        # Primitive estimator
        self.estimator = StatevectorEstimator()

    def _build_circuit(self) -> QuantumCircuit:
        """Create a parameter‑ized ansatz with data encoding and entanglement."""
        qc = QuantumCircuit(self.num_qubits)
        # Data encoding (RX gates)
        for q, param in enumerate(self.input_params):
            qc.rx(param, q)
        # Entangling layers
        for d in range(self.depth):
            for q in range(self.num_qubits):
                idx = d * self.num_qubits + q
                qc.ry(self.weight_params[idx], q)
                qc.rz(self.weight_params[idx], q)
            # Entangle neighbours (cyclic)
            for q in range(self.num_qubits):
                qc.cx(q, (q + 1) % self.num_qubits)
        return qc

    def get_estimator_qnn(self) -> QiskitEstimatorQNN:
        """
        Wrap the circuit in a Qiskit EstimatorQNN for use with
        gradient‑based optimizers.
        """
        return QiskitEstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

__all__ = ["EstimatorQNN"]
