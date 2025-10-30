"""Quantum estimator that embeds a convolution‑like sub‑circuit.

The circuit first encodes the input data via a set of RX gates whose
parameters are tied to the pixel values of a 2‑D kernel.  A second
parameterized layer (RY gates) serves as the trainable weights.  The
expectation value of a Pauli‑Y operator applied to every qubit (tensor
product) is used as the output observable, matching the structure of
the classical estimator.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator


class EstimatorQNNHybrid:
    """Hybrid quantum estimator that mirrors the classical EstimatorQNNHybrid.

    Attributes
    ----------
    kernel_size : int
        Size of the 2‑D kernel (default 2 → 4 qubits).
    backend : qiskit.providers.Backend
        Backend used for statevector simulation.
    shots : int
        Number of shots for measurement‑based estimation.
    threshold : float
        Threshold used to binarise input data before encoding.
    circuit : qiskit.circuit.QuantumCircuit
        Full variational circuit comprising a data‑encoding layer and a
        trainable rotation layer.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        backend: qiskit.providers.Backend | None = None,
        shots: int = 100,
        threshold: float = 127.0,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.backend = backend or qiskit.Aer.get_backend("statevector_simulator")
        self.shots = shots
        self.threshold = threshold
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        """Create a circuit with a data‑encoding layer followed by a
        trainable rotation layer.
        """
        qc = QuantumCircuit(self.n_qubits)

        # Data‑encoding layer (RX gates)
        self.data_params = [
            Parameter(f"theta{i}") for i in range(self.n_qubits)
        ]
        for i, p in enumerate(self.data_params):
            qc.rx(p, i)

        # Optional entanglement
        qc.barrier()
        qc += random_circuit(self.n_qubits, 2)

        # Trainable rotation layer (RY gates)
        self.weight_params = [
            Parameter(f"w{i}") for i in range(self.n_qubits)
        ]
        for i, p in enumerate(self.weight_params):
            qc.ry(p, i)

        return qc

    def get_estimator(self) -> EstimatorQNN:
        """Return a qiskit Machine Learning EstimatorQNN instance."""
        # Observable: tensor product of Y on all qubits
        observable = SparsePauliOp.from_list(
            [(("Y" * self.n_qubits), 1)]
        )

        estimator = StatevectorEstimator(backend=self.backend)

        return EstimatorQNN(
            circuit=self.circuit,
            observables=observable,
            input_params=self.data_params,
            weight_params=self.weight_params,
            estimator=estimator,
        )


__all__ = ["EstimatorQNNHybrid"]
