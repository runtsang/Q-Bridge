"""Quantum classifier that extends the original data‑re‑uploading ansatz
with a classical post‑processing layer and a lightweight training routine.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import AerSimulator

class QuantumClassifierModel:
    """
    Hybrid quantum‑classical classifier.

    The model builds a data‑re‑uploading ansatz, evaluates expectation
    values of Pauli‑Z observables, and feeds them into a small linear
    layer. Training uses the parameter‑shift rule for the quantum part
    and a conventional gradient‑step for the post‑processing weights.
    """

    def __init__(self,
                 num_qubits: int,
                 depth: int = 2,
                 entanglement: str = "cnot",
                 seed: int | None = None):
        self.num_qubits = num_qubits
        self.depth = depth
        self.entanglement = entanglement
        # quantum backend
        self.backend = QuantumInstance(
            backend=AerSimulator(method="statevector"),
            seed_simulator=seed,
            seed_transpiler=seed,
        )
        # circuit, encoding, variational weights, observables
        self.circuit, self.encoding, self.weights, self.observables = self.build_classifier_circuit(
            num_qubits, depth, entanglement
        )
        # initialise variational parameters uniformly in [0, 2π)
        self.param_values = np.random.uniform(0, 2 * np.pi, len(self.weights))
        # classical post‑processing: 2‑class logits from observable vector
        self.post_layer = np.random.randn(2, len(self.observables))

    @staticmethod
    def build_classifier_circuit(num_qubits: int,
                                 depth: int,
                                 entanglement: str = "cnot") -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        """
        Construct a layered data‑re‑uploading ansatz.

        Returns
        -------
        circuit : QuantumCircuit
            Parameterised circuit.
        encoding : list
            List of data‑encoding parameters.
        weights : list
            List of variational parameters.
        observables : list
            Pauli‑Z observables for each qubit.
        """
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)

        circuit = QuantumCircuit(num_qubits)

        # Data encoding (Rx)
        for param, qubit in zip(encoding, range(num_qubits)):
            circuit.rx(param, qubit)

        # Variational layers with entanglement
        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                circuit.ry(weights[idx], qubit)
                idx += 1
            if entanglement == "cnot":
                for qubit in range(num_qubits - 1):
                    circuit.cx(qubit, qubit + 1)
            elif entanglement == "cz":
                for qubit in range(num_qubits - 1):
                    circuit.cz(qubit, qubit + 1)
            else:  # nearest‑neighbour default
                for qubit in range(num_qubits - 1):
                    circuit.cx(qubit, qubit + 1)

        observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                       for i in range(num_qubits)]
        return circuit, list(encoding), list(weights), observables

    def _state_expectation(self, params: np.ndarray, data: np.ndarray, obs: SparsePauliOp) -> float:
        """
        Exact expectation value of a Pauli observable on the state produced by the
        circuit with given variational parameters and data.
        """
        circ = self.circuit.copy()
        # bind variational parameters
        circ.assign_parameters({str(p): v for p, v in zip(self.weights, params)}, inplace=True)
        # bind data
        circ.assign_parameters({str(p): v for p, v in zip(self.encoding, data)}, inplace=True)
        result = self.backend.run(circ, shots=0)
        state = result.get_statevector()
        op_mat = obs.to_matrix()
        return np.real(state.conj().T @ op_mat @ state)

    def train(self,
              X: np.ndarray,
              y: np.ndarray,
              lr: float = 0.01,
              epochs: int = 20,
              verbose: bool = False) -> None:
        """
        Very lightweight training routine that optimises the variational parameters
        via the parameter‑shift rule and the post‑processing weights by stochastic
        gradient descent.
        """
        for epoch in range(epochs):
            epoch_loss = 0.0
            for xi, yi in zip(X, y):
                # Expectation vector for current parameters
                exps = np.array([self._state_expectation(self.param_values, xi, obs)
                                 for obs in self.observables])
                # Logits
                logits = self.post_layer @ exps
                probs = np.exp(logits) / np.sum(np.exp(logits))
                # Cross‑entropy loss
                loss = -np.log(probs[yi] + 1e-12)
                epoch_loss += loss

                # Gradients for logits
                grad_logits = probs
                grad_logits[yi] -= 1

                # Gradient for post‑processing layer
                grad_post = np.outer(grad_logits, exps)
                # Update post‑processing weights
                self.post_layer -= lr * grad_post

                # Gradient for variational parameters
                grad_exps = self.post_layer.T @ grad_logits
                for i in range(len(self.param_values)):
                    shift = np.pi / 2
                    pos = self.param_values.copy()
                    neg = self.param_values.copy()
                    pos[i] += shift
                    neg[i] -= shift
                    exp_pos = np.array([self._state_expectation(pos, xi, obs)
                                        for obs in self.observables])
                    exp_neg = np.array([self._state_expectation(neg, xi, obs)
                                        for obs in self.observables])
                    grad_params_i = np.dot(grad_exps, (exp_pos - exp_neg) / 2)
                    self.param_values[i] -= lr * grad_params_i
            if verbose:
                print(f"[QML] Epoch {epoch+1}/{epochs} – loss: {epoch_loss/len(X):.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return predicted class labels (0 or 1).
        """
        preds = []
        for xi in X:
            exps = np.array([self._state_expectation(self.param_values, xi, obs)
                             for obs in self.observables])
            logits = self.post_layer @ exps
            preds.append(np.argmax(logits))
        return np.array(preds)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Accuracy on the provided dataset.
        """
        preds = self.predict(X)
        return np.mean(preds == y)
