"""Hybrid quantum fully‑connected layer with input and weight parameters.

This module defines a quantum circuit that can be used as a
parameterized layer in a hybrid classical‑quantum model.  It
combines the simple H‑RY circuit of the original FCL example with
the EstimatorQNN interface from the second reference, providing
a state‑vector estimator and a Y‑observable over multiple qubits.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator

class FCLImpl:
    """
    Quantum surrogate for a fully‑connected layer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit. Default is 2.
    shots : int
        Number of shots for the qasm simulator. Default 1000.
    """

    def __init__(self, n_qubits: int = 2, shots: int = 1000) -> None:
        # Define input and weight parameters
        self.input_params = [Parameter(f"inp_{i}") for i in range(n_qubits)]
        self.weight_params = [Parameter(f"w_{i}") for i in range(n_qubits)]

        # Build a simple variational circuit
        self.circuit = QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        for q in range(n_qubits):
            self.circuit.ry(self.input_params[q], q)
            self.circuit.rz(self.weight_params[q], q)
        self.circuit.measure_all()

        # Observable: sum of Y on all qubits
        self.observable = SparsePauliOp.from_list([("Y" * n_qubits, 1)])

        # Estimator for expectation values
        backend = Aer.get_backend("statevector_simulator")
        self.estimator = StatevectorEstimator(backend=backend)
        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def run(self, thetas: np.ndarray | list[float]) -> np.ndarray:
        """
        Execute the circuit with the supplied parameters.

        Parameters
        ----------
        thetas : array‑like
            Concatenated input and weight parameters: first n_qubits
            belong to input_params, the rest to weight_params.

        Returns
        -------
        np.ndarray
            Expectation value of the observable as a 1‑element array.
        """
        thetas = np.asarray(thetas, dtype=float)
        if thetas.size!= len(self.input_params) + len(self.weight_params):
            raise ValueError(
                f"Expected {len(self.input_params) + len(self.weight_params)} parameters, "
                f"got {thetas.size}"
            )
        param_dict = {
            **{p: v for p, v in zip(self.input_params, thetas[: len(self.input_params)])},
            **{p: v for p, v in zip(self.weight_params, thetas[len(self.input_params) :])},
        }
        # Evaluate expectation via EstimatorQNN
        expectation = self.estimator_qnn(param_dict)
        # EstimatorQNN returns a list of expectation values; we take the first.
        return np.array([expectation[0]])

def FCL() -> type[FCLImpl]:
    """Return the FCLImpl class for compatibility with the anchor API."""
    return FCLImpl

__all__ = ["FCLImpl", "FCL"]
