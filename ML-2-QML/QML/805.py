"""Advanced quantum regression estimator using Qiskit."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator

class AdvancedEstimatorQNN:
    """
    Quantum neural network for regression that expands upon the original
    EstimatorQNN example.

    Features:
    - 2 qubits with a 2‑layer parameterised circuit.
    - Entanglement via CX gates between layers.
    - Input parameters encoded as Ry rotations.
    - Weight parameters applied through RX rotations.
    - Expectation value of a two‑qubit observable as output.
    """

    def __init__(
        self,
        n_qubits: int = 2,
        n_layers: int = 2,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.circuit = self._build_circuit()
        self.observable = self._build_observable()
        self.estimator = StatevectorEstimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=[self.circuit.parameters[0]],
            weight_params=self.circuit.parameters[1:],
            estimator=self.estimator,
        )

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        # Input encoding
        qc.ry(Parameter("x"), 0)
        qc.ry(Parameter("x"), 1)
        # Parameterised layers
        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                qc.rx(Parameter(f"w_{layer}_{qubit}"), qubit)
            # Entanglement
            if self.n_qubits > 1:
                qc.cx(0, 1)
        return qc

    def _build_observable(self) -> SparsePauliOp:
        # Use Z⊗Z as a simple observable
        return SparsePauliOp.from_list([("ZZ", 1)])

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the QNN on a batch of inputs.

        Parameters
        ----------
        x : array-like, shape (n_samples,)
            Input values to be encoded.

        Returns
        -------
        ndarray
            Predicted output values.
        """
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        predictions = []
        for val in x:
            params = {self.circuit.parameters[0]: val}
            # Random initial weights for demonstration; in practice
            # these would be optimised.
            for param in self.circuit.parameters[1:]:
                params[param] = np.random.uniform(0, 2 * np.pi)
            result = self.estimator.run(
                circuits=[self.circuit],
                parameter_values=[params]
            ).result()
            predictions.append(result.values[0][0])
        return np.array(predictions)

__all__ = ["AdvancedEstimatorQNN"]
