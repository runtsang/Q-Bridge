from __future__ import annotations

import numpy as np
import torch
from qiskit.providers.aer import AerSimulator
from qiskit.algorithms.optimizers import COBYLA
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import ParameterVector
from QuantumKernelMethod import Kernel as QuantumKernel, kernel_matrix as quantum_kernel_matrix
from QuantumClassifierModel import build_classifier_circuit

class HybridKernelClassifier:
    """Quantum‑kernel + variational‑circuit classifier.

    The kernel is evaluated with a TorchQuantum ansatz; the classifier
    is a Qiskit circuit with encoding and variational layers and
    Z‑measurements used as logits.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / input dimensionality.
    depth : int, default 2
        Depth of the variational layers.
    """
    def __init__(self, num_qubits: int, depth: int = 2) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        # Quantum kernel
        self.kernel = QuantumKernel()
        # Quantum classifier circuit
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(
            num_qubits, depth
        )
        self.backend = AerSimulator()
        self.params = np.zeros(len(self.weights), dtype=float)
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Optimize variational parameters using a classical optimizer."""
        self.X_train = X
        self.y_train = y
        optimizer = COBYLA(maxiter=200)

        def loss_fn(p: np.ndarray) -> float:
            self.params = p
            preds = self.predict(X)
            return np.mean((preds - y) ** 2)

        optimizer.optimize(num_vars=len(self.params), objective_function=loss_fn)
        self.params = optimizer.params

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels using the trained variational circuit."""
        if self.X_train is None:
            raise RuntimeError("Model has not been fitted.")
        # Compute quantum kernel Gram matrix (kept for API compatibility)
        K = quantum_kernel_matrix(X, self.X_train)
        preds = []
        for x in X:
            # Build parameter dictionary
            param_dict = {var: val for var, val in zip(self.weights, self.params)}
            for idx, enc in enumerate(self.encoding):
                param_dict[enc] = x[idx]
            bound_circuit = self.circuit.bind_parameters(param_dict)
            job = self.backend.run(bound_circuit, shots=1024)
            result = job.result()
            # Expectation values of Z observables
            exp_vals = []
            for obs in self.observables:
                exp = result.get_expectation_value(obs, bound_circuit)
                exp_vals.append(exp)
            # Simple threshold on the sum of expectations
            pred = 1 if sum(exp_vals) > 0 else 0
            preds.append(pred)
        return np.array(preds)
