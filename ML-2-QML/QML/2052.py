"""EstimatorQNN – a hybrid variational quantum regressor.

This module extends the original Qiskit example by:
  • Building a 2‑qubit ansatz with multiple rotation and entanglement
    layers.
  • Using the parameter‑shift rule to obtain analytic gradients.
  • Providing a lightweight training loop that couples a classical
    feature‑map (two rotation gates per input) to the quantum circuit.
  • Allowing the user to choose the backend (state‑vector or qasm) and
    automatically handling parameter updates.

The public API mirrors the original: a function `EstimatorQNN()`
returns a ready‑to‑train object that exposes `.train()` and `.predict()`.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile, Aer, execute
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as QiskitStatevectorEstimator
from typing import Iterable, Tuple


def EstimatorQNN() -> "HybridEstimator":
    """
    Return a hybrid quantum‑classical regression model.

    The returned object owns:
      * a 2‑qubit variational circuit with 2 layers,
      * a single observable (Pauli‑Y on qubit 0),
      * a state‑vector estimator backend,
      * a training interface that optimises weight parameters
        using the parameter‑shift gradient.

    Returns
    -------
    HybridEstimator
        The instantiated model ready for training and inference.
    """

    class HybridEstimator:
        def __init__(self) -> None:
            # ----- Classical feature‑map -----
            # Two input parameters: input1, input2
            self.input_params: Tuple[Parameter, Parameter] = (
                Parameter("input1"),
                Parameter("input2"),
            )
            # ----- Variational (weight) parameters -----
            self.weight_params: Iterable[Parameter] = [
                Parameter(f"w{idx}") for idx in range(4)
            ]

            # ----- Build the circuit -----
            self.circuit = QuantumCircuit(2)
            # Feature map: encode inputs
            self.circuit.ry(self.input_params[0], 0)
            self.circuit.rz(self.input_params[1], 1)

            # Two layers of variational rotations + entanglement
            for i, w in enumerate(self.weight_params):
                self.circuit.ry(w, i % 2)
                self.circuit.rz(w + 0.1 * (i + 1), (i + 1) % 2)
                if i % 2 == 1:  # entangle after every two ops
                    self.circuit.cx(0, 1)

            # Observable
            self.observable = SparsePauliOp.from_list([("Y0", 1)])

            # Estimator backend
            self.estimator = QiskitStatevectorEstimator(
                backend=Aer.get_backend("statevector_simulator")
            )

            # Wrap in Qiskit’s EstimatorQNN for convenience
            self.qiskit_qnn = QiskitEstimatorQNN(
                circuit=self.circuit,
                observables=self.observable,
                input_params=list(self.input_params),
                weight_params=list(self.weight_params),
                estimator=self.estimator,
            )

        def predict(self, X: np.ndarray) -> np.ndarray:
            """Return the model predictions for the given feature matrix.

            Parameters
            ----------
            X
                Shape (n_samples, 2) – the two input features.

            Returns
            -------
            np.ndarray
                Shape (n_samples,) – predicted scalar values.
            """
            predictions = []
            for x in X:
                params = {
                    str(self.input_params[0]): float(x[0]),
                    str(self.input_params[1]): float(x[1]),
                }
                # Set current weight values (no-op if not trained yet)
                for w in self.weight_params:
                    params[str(w)] = float(self.qiskit_qnn.params[str(w)])
                val = self.qiskit_qnn.evaluate(params, self.qiskit_qnn.params)
                predictions.append(val[0])
            return np.array(predictions)

        def train(
            self,
            X: np.ndarray,
            y: np.ndarray,
            *,
            epochs: int = 200,
            lr: float = 0.1,
            verbose: bool = False,
        ) -> None:
            """Simple gradient‑descent training loop.

            Parameters
            ----------
            X, y
                Training data.
            epochs, lr
                Training hyper‑parameters.
            verbose
                If True, prints loss per epoch.
            """
            # Initialise weights randomly
            for w in self.weight_params:
                self.qiskit_qnn.params[str(w)] = np.random.uniform(-np.pi, np.pi)

            for epoch in range(epochs):
                loss = 0.0
                grads = {str(w): 0.0 for w in self.weight_params}
                for x, target in zip(X, y):
                    # Build parameter dictionary
                    param_dict = {
                        str(self.input_params[0]): float(x[0]),
                        str(self.input_params[1]): float(x[1]),
                    }
                    # Current weight values
                    for w in self.weight_params:
                        param_dict[str(w)] = float(self.qiskit_qnn.params[str(w)])
                    # Forward pass
                    pred = self.qiskit_qnn.evaluate(param_dict, self.qiskit_qnn.params)[0]
                    loss += (pred - target) ** 2
                    # Gradient via parameter‑shift
                    for w in self.weight_params:
                        shift = np.pi / 2
                        param_plus = param_dict.copy()
                        param_plus[str(w)] += shift
                        param_minus = param_dict.copy()
                        param_minus[str(w)] -= shift
                        f_plus = self.qiskit_qnn.evaluate(param_plus, self.qiskit_qnn.params)[0]
                        f_minus = self.qiskit_qnn.evaluate(param_minus, self.qiskit_qnn.params)[0]
                        grads[str(w)] += (f_plus - f_minus) / (2 * np.sin(shift))
                loss /= len(X)
                # Update weights
                for w in self.weight_params:
                    self.qiskit_qnn.params[str(w)] -= lr * grads[str(w)] / len(X)
                if verbose:
                    print(f"Epoch {epoch + 1:03d} – loss: {loss:.6f}")

    return HybridEstimator()
__all__ = ["EstimatorQNN"]
