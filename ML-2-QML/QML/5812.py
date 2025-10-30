"""Core quantum circuit factory for an incremental data‑uploading classifier with hybrid training."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class QuantumClassifierModel:
    """
    Variational quantum classifier that mirrors the classical API.
    The circuit consists of an encoding layer followed by a configurable
    depth of Ry rotations and a tunable entanglement pattern.  The class
    exposes ``train_step`` and ``predict`` that operate on PyTorch tensors
    and use a statevector simulator to obtain expectation values.
    """
    def __init__(self, num_qubits: int, depth: int = 2,
                 entanglement: str = "full",
                 backend: str = "statevector_simulator") -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.entanglement = entanglement
        self.backend = backend

        # Build the parameterised circuit
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()
        # Parameter vector for training
        self.params = np.concatenate([self.encoding, self.weights])
        # Optimiser placeholder
        self.optimizer = None

    def _build_circuit(self) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)
        for q, param in enumerate(encoding):
            qc.rx(param, q)

        idx = 0
        for _ in range(self.depth):
            for q in range(self.num_qubits):
                qc.ry(weights[idx], q)
                idx += 1
            # entanglement pattern
            if self.entanglement == "full" or self.entanglement == "linear":
                for q in range(self.num_qubits - 1):
                    qc.cz(q, q + 1)
            # else: no entanglement

        observables = [SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
                       for i in range(self.num_qubits)]
        return qc, list(encoding), list(weights), observables

    def _expectation(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Compute expectation values of the observables for a single data point."""
        # Bind parameters
        bound_circuit = self.circuit.bind_parameters(
            {param: val for param, val in zip(self.encoding + self.weights, params)}
        )
        # Replace encoding parameters with data
        for i, val in enumerate(x):
            bound_circuit.data[i][1] = val  # modify rx angle

        job = execute(bound_circuit, Aer.get_backend(self.backend), shots=1024)
        result = job.result()
        exp_vals = []
        for op in self.observables:
            exp = result.get_expectation_value(op.to_matrix(), bound_circuit)
            exp_vals.append(exp)
        return np.array(exp_vals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Batch forward pass returning logits."""
        batch_size = x.shape[0]
        logits = torch.zeros(batch_size, 2)
        for i in range(batch_size):
            exp = self._expectation(self.params, x[i].numpy())
            logits[i, 0] = 1 - exp.sum()  # dummy mapping to two classes
            logits[i, 1] = exp.sum()
        return logits

    def train_step(self, x: torch.Tensor, y: torch.Tensor,
                   optimizer: torch.optim.Optimizer,
                   loss_fn: nn.Module = nn.CrossEntropyLoss()) -> torch.Tensor:
        self.eval()
        optimizer.zero_grad()
        logits = self.forward(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        return loss

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        return (probs[:, 1] > threshold).long()


def build_classifier_circuit(num_qubits: int, depth: int = 2) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Backwards‑compatible helper that returns the underlying circuit and metadata."""
    model = QuantumClassifierModel(num_qubits, depth)
    return model.circuit, model.encoding, model.weights, model.observables


__all__ = ["QuantumClassifierModel", "build_classifier_circuit"]
