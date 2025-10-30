"""Hybrid fully‑connected layer implemented as a parameterised quantum circuit.

The quantum implementation follows the EstimatorQNN seed but extends it with
multiple qubits and a two‑layer ansatz.  Each input and weight parameter
is mapped to a distinct rotation, and the circuit is wrapped in
``qiskit_machine_learning.neural_networks.EstimatorQNN`` to provide a
high‑level :py:meth:`run` interface that returns a NumPy array of the
expectation value.  The class is named ``HybridLayer`` to match the
classical counterpart, enabling side‑by‑side experimentation.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator


class HybridLayer:
    """
    Quantum hybrid layer that maps a 4‑parameter input vector to a single
    expectation value using a two‑qubit variational circuit.
    """
    def __init__(self, n_qubits: int = 2, backend=None, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self._build_circuit()

    def _build_circuit(self) -> None:
        # Define input and weight parameters
        self.input_params = [Parameter(f"input{i}") for i in range(self.n_qubits)]
        self.weight_params = [Parameter(f"weight{i}") for i in range(self.n_qubits)]

        # Build a simple two‑layer ansatz
        self.circuit = QuantumCircuit(self.n_qubits)
        for q in range(self.n_qubits):
            self.circuit.h(q)
            self.circuit.ry(self.input_params[q], q)
            self.circuit.rx(self.weight_params[q], q)
        # Entangling layer
        self.circuit.cx(0, 1)
        self.circuit.measure_all()

        # Observable (here Y on all qubits)
        self.observable = SparsePauliOp.from_list([("Y" * self.n_qubits, 1)])

        # Estimator wrapper
        self.estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def run(self, thetas: list[float]) -> np.ndarray:
        """
        Evaluate the quantum circuit for a list of four parameters.

        Parameters
        ----------
        thetas : list[float]
            Length must be 2 * n_qubits.  The first half are mapped to
            input parameters, the second half to weight parameters.
        """
        if len(thetas)!= len(self.input_params) + len(self.weight_params):
            raise ValueError("Incorrect number of parameters for the hybrid layer.")

        # Build the parameter bind dictionary
        bind_dict = {
            **{p: t for p, t in zip(self.input_params, thetas[:len(self.input_params)])},
            **{p: t for p, t in zip(self.weight_params, thetas[len(self.input_params):])},
        }

        # Execute the circuit using the EstimatorQNN wrapper
        expectation = self.estimator_qnn.run(bind_dict, shots=self.shots)
        # The EstimatorQNN returns an array of shape (1,) with the expectation
        return np.array([expectation])


def FCL(n_qubits: int = 2, backend=None, shots: int = 1024) -> HybridLayer:
    """
    Factory function mirroring the original FCL API for the quantum
    implementation.  Users can supply a backend and shot number if
    desired; defaults use the local Aer simulator.
    """
    if backend is None:
        from qiskit import Aer
        backend = Aer.get_backend("qasm_simulator")
    return HybridLayer(n_qubits=n_qubits, backend=backend, shots=shots)


__all__ = ["HybridLayer", "FCL"]
