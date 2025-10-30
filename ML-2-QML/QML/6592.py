from __future__ import annotations

from typing import List, Tuple

import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class QuantumClassifierModel:
    """Variational quantum classifier with data‑re‑uploading ansatz and hybrid classical read‑out."""
    def __init__(self, num_qubits: int, depth: int, backend=None):
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = backend or Aer.get_backend('statevector_simulator')
        self.encoding = ParameterVector("x", num_qubits)
        self.weight_params = ParameterVector("theta", num_qubits * depth)
        self.circuit, self.observables = self._build_circuit()
        self.weights = np.random.uniform(-np.pi, np.pi, size=len(self.weight_params))
        self.classical_weights = np.random.randn(num_qubits)
        self.classical_bias = 0.0

    def _build_circuit(self) -> Tuple[QuantumCircuit, List[SparsePauliOp]]:
        circuit = QuantumCircuit(self.num_qubits)
        for q, param in enumerate(self.encoding):
            circuit.rx(param, q)
        idx = 0
        for _ in range(self.depth):
            for q in range(self.num_qubits):
                circuit.ry(self.weight_params[idx], q)
                idx += 1
            for q in range(self.num_qubits - 1):
                circuit.cz(q, q + 1)
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]
        return circuit, observables

    def _expectation_vector(self, data_point: np.ndarray) -> np.ndarray:
        qc = self.circuit.copy()
        binding = {self.encoding[i]: data_point[i] for i in range(self.num_qubits)}
        binding.update(
            {self.weight_params[i]: self.weights[i] for i in range(len(self.weight_params))}
        )
        qc.bind_parameters(binding)
        sv = execute(qc, self.backend, shots=0).result().get_statevector()
        exp_vals = []
        for q in range(self.num_qubits):
            exp = 0.0
            for idx, amp in enumerate(sv):
                if ((idx >> q) & 1) == 0:
                    exp += abs(amp) ** 2
                else:
                    exp -= abs(amp) ** 2
            exp_vals.append(exp)
        return np.array(exp_vals)

    def expectation_values(self, data: np.ndarray) -> np.ndarray:
        return np.array([self._expectation_vector(x) for x in data])

    def cost(self, data: np.ndarray, labels: np.ndarray) -> float:
        exp_vals = self.expectation_values(data)
        logits = exp_vals @ self.classical_weights + self.classical_bias
        probs = 1 / (1 + np.exp(-logits))
        eps = 1e-12
        loss = -np.mean(
            labels * np.log(probs + eps) + (1 - labels) * np.log(1 - probs + eps)
        )
        return loss

    def train(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01,
        epochs: int = 10,
        verbose: bool = False,
    ) -> None:
        for epoch in range(epochs):
            loss = self.cost(data, labels)
            grad_q = np.zeros_like(self.weights)
            shift = np.pi / 2
            for i in range(len(self.weights)):
                orig = self.weights[i]
                self.weights[i] = orig + shift
                loss_plus = self.cost(data, labels)
                self.weights[i] = orig - shift
                loss_minus = self.cost(data, labels)
                grad_q[i] = (loss_plus - loss_minus) / (2 * np.sin(shift))
                self.weights[i] = orig
            exp_vals = self.expectation_values(data)
            logits = exp_vals @ self.classical_weights + self.classical_bias
            probs = 1 / (1 + np.exp(-logits))
            grad_cw = -np.mean((labels - probs)[:, None] * exp_vals, axis=0)
            grad_cb = -np.mean(labels - probs)
            self.weights -= lr * grad_q
            self.classical_weights -= lr * grad_cw
            self.classical_bias -= lr * grad_cb
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} | loss: {loss:.4f}")

    def predict(self, data: np.ndarray) -> np.ndarray:
        exp_vals = self.expectation_values(data)
        logits = exp_vals @ self.classical_weights + self.classical_bias
        probs = 1 / (1 + np.exp(-logits))
        return (probs >= 0.5).astype(int)

    def get_encoding(self) -> List[int]:
        return list(range(self.num_qubits))

    def get_weights(self) -> np.ndarray:
        return self.weights.copy()

    def get_observables(self) -> List[SparsePauliOp]:
        return self.observables


__all__ = ["QuantumClassifierModel"]
