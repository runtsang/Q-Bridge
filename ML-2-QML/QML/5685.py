"""
HybridClassifier for variational quantum classification using Qiskit.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize
from sklearn.preprocessing import OneHotEncoder


class HybridClassifier:
    """
    Variational quantum classifier with a parameter‑shift training loop.
    Supports multiple backends and configurable depth.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int = 3,
        backend: str = "aer_simulator_statevector",
        shots: int = 1024,
        max_grad_norm: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        num_qubits : int
            Number of qubits / input features.
        depth : int
            Number of ansatz layers.
        backend : str
            Qiskit backend name.
        shots : int
            Number of shots for expectation estimation.
        max_grad_norm : float
            Gradient clipping threshold.
        """
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = backend
        self.shots = shots
        self.max_grad_norm = max_grad_norm

        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()
        self.param_bounds = [(0, 2 * np.pi)] * len(self.weights)
        self.backend_obj = Aer.get_backend(self.backend)

    def _build_circuit(self) -> Tuple[QuantumCircuit, ParameterVector, ParameterVector, List[SparsePauliOp]]:
        """
        Construct a layered ansatz with RX encoding and CZ entanglement.
        """
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)
        for param, qubit in zip(encoding, range(self.num_qubits)):
            qc.rx(param, qubit)

        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]
        return qc, encoding, weights, observables

    def _expectation(self, circuit: QuantumCircuit, params: np.ndarray) -> np.ndarray:
        """
        Evaluate expectation values of the observables for given parameters.
        """
        bound_circuit = circuit.bind_parameters(
            {param: val for param, val in zip(self.weights, params)}
        )
        job = execute(bound_circuit, self.backend_obj, shots=self.shots)
        result = job.result()
        state = result.get_statevector(bound_circuit)
        exp_vals = np.array(
            [state.expectation_value(obs).real for obs in self.observables]
        )
        return exp_vals

    def _parameter_shift_gradient(self, params: np.ndarray) -> np.ndarray:
        """
        Compute gradient via the parameter‑shift rule.
        """
        grad = np.zeros_like(params)
        shift = np.pi / 2.0
        for i in range(len(params)):
            shifted_pos = params.copy()
            shifted_neg = params.copy()
            shifted_pos[i] += shift
            shifted_neg[i] -= shift
            exp_pos = self._expectation(self.circuit, shifted_pos)
            exp_neg = self._expectation(self.circuit, shifted_neg)
            grad[i] = (exp_pos - exp_neg) / 2.0
        return grad

    def _loss_and_grad(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute cross‑entropy loss and its gradient over the dataset.
        """
        n_samples = X.shape[0]
        probs = np.zeros((n_samples, 2))
        for idx, sample in enumerate(X):
            params_sample = params.copy()
            params_sample[: self.num_qubits] = sample  # encode data into first layer
            exp_vals = self._expectation(self.circuit, params_sample)
            probs[idx] = self._softmax(exp_vals)

        # One‑hot encode labels
        encoder = OneHotEncoder(sparse=False, categories="auto")
        y_onehot = encoder.fit_transform(y.reshape(-1, 1))
        loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-12), axis=1))

        grad = np.zeros_like(params)
        for idx, sample in enumerate(X):
            params_sample = params.copy()
            params_sample[: self.num_qubits] = sample
            exp_vals = self._expectation(self.circuit, params_sample)
            probs_sample = self._softmax(exp_vals)
            grad_sample = self._parameter_shift_gradient(params_sample)
            grad += (probs_sample - y_onehot[idx])[:, None] * grad_sample
        grad /= n_samples
        # Gradient clipping
        grad_norm = np.linalg.norm(grad)
        if grad_norm > self.max_grad_norm:
            grad = grad * self.max_grad_norm / grad_norm
        return loss, grad

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        verbose: bool = True,
    ) -> None:
        """
        Train the variational circuit using L‑BFGS‑B optimizer.

        Parameters
        ----------
        X : np.ndarray
            Binary input data of shape (n_samples, num_qubits).
        y : np.ndarray
            Binary labels (0 or 1) of shape (n_samples,).
        epochs : int
            Number of optimization iterations.
        verbose : bool
            Whether to print progress.
        """
        # Initialize parameters randomly
        init_params = np.random.uniform(0, 2 * np.pi, size=len(self.weights))

        def objective(params):
            loss, grad = self._loss_and_grad(params, X, y)
            return loss, grad

        res = minimize(
            objective,
            init_params,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": epochs, "disp": verbose},
            bounds=self.param_bounds,
        )
        self.params = res.x

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input data.

        Parameters
        ----------
        X : np.ndarray
            Binary input data of shape (n_samples, num_qubits).

        Returns
        -------
        np.ndarray
            Predicted labels (0 or 1).
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for input data.

        Parameters
        ----------
        X : np.ndarray
            Binary input data of shape (n_samples, num_qubits).

        Returns
        -------
        np.ndarray
            Probabilities of shape (n_samples, 2).
        """
        n_samples = X.shape[0]
        probs = np.zeros((n_samples, 2))
        for idx, sample in enumerate(X):
            params = self.params.copy()
            params[: self.num_qubits] = sample
            exp_vals = self._expectation(self.circuit, params)
            probs[idx] = self._softmax(exp_vals)
        return probs

    @staticmethod
    def build_classifier_circuit(
        num_qubits: int,
        depth: int,
    ) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
        """
        Construct a circuit and return its components for interface compatibility.

        Parameters
        ----------
        num_qubits : int
            Number of qubits.
        depth : int
            Depth of the ansatz.

        Returns
        -------
        circuit : QuantumCircuit
            The constructed circuit.
        encoding : Iterable[ParameterVector]
            List containing the encoding ParameterVector.
        weights : Iterable[ParameterVector]
            List containing the weight ParameterVector.
        observables : List[SparsePauliOp]
            The observable operators.
        """
        instance = HybridClassifier(num_qubits, depth)
        return instance.circuit, [instance.encoding], [instance.weights], instance.observables


__all__ = ["HybridClassifier"]
