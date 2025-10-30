"""Extended quantum neural network with multi‑qubit variational circuit.

The module implements the ExtendedEstimatorQNN class that extends the
original Qiskit EstimatorQNN by:
- Using a 3‑qubit entangled circuit.
- Parameterising rotations with both data and trainable weights.
- Employing a PauliZ⊗Z⊗Z observable for collective read‑out.
- Providing a ``predict`` method that accepts a NumPy input array and
  returns expectation values as a NumPy array.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator

class ExtendedEstimatorQNN:
    """Quantum neural network wrapper with a 3‑qubit entangled ansatz."""

    def __init__(
        self,
        input_dim: int = 2,
        seed: int | None = None,
    ) -> None:
        if seed is not None:
            np.random.seed(seed)

        # Define symbolic parameters: first for data, second for trainable weights
        data_params = [Parameter(f"x{i}") for i in range(input_dim)]
        weight_params = [Parameter(f"w{i}") for i in range(2 * input_dim)]

        # Build a 3‑qubit circuit with entanglement and layers of rotations
        qc = QuantumCircuit(3)
        # Data encoding on first two qubits
        for i, param in enumerate(data_params):
            qc.ry(param, i)
        # Entanglement
        qc.cx(0, 1)
        qc.cx(1, 2)
        # Parameterised layers
        for param in weight_params[:input_dim]:
            qc.rx(param, 0)
        for param in weight_params[input_dim:]:
            qc.ry(param, 1)

        # Observable: Z⊗Z⊗Z
        observable = SparsePauliOp.from_list([("ZZZ", 1)])

        # Create the Qiskit EstimatorQNN instance
        self.estimator_qnn = EstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=data_params,
            weight_params=weight_params,
            estimator=StatevectorEstimator(),
        )

        self.input_dim = input_dim
        self.weight_params = weight_params

    def predict(self, X: np.ndarray | list[float]) -> np.ndarray:
        """Evaluate the variational circuit on the given data points."""
        # Optional: configure shots or backend options
        self.estimator_qnn.estimator.set_options(shots=1024)
        # Ensure X has shape (n_samples, input_dim)
        X = np.atleast_2d(X)
        results = self.estimator_qnn.predict(X)
        return results.squeeze()

    def set_weights(self, weights: np.ndarray) -> None:
        """Assign trainable weights to the circuit."""
        if weights.shape[0]!= len(self.weight_params):
            raise ValueError("Incorrect number of weight parameters.")
        param_dict = {p: w for p, w in zip(self.weight_params, weights)}
        self.estimator_qnn.circuit.assign_parameters(param_dict, inplace=True)

__all__ = ["ExtendedEstimatorQNN"]
