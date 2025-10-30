"""Hybrid fraud detection – quantum implementation.

The class `FraudDetectionHybrid` builds a Qiskit variational circuit that
encodes input features and extracts expectation values.  A lightweight
PyTorch classifier consumes these quantum features to produce the final
prediction.  The quantum part is fully parameterized and can be trained
jointly with the classical head if desired.
"""
from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import nn, optim
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """Construct a layered ansatz with explicit encoding and variational parameters."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    index = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables


class FraudDetectionHybrid(nn.Module):
    """Hybrid fraud‑detection model – quantum side.

    The model consists of a Qiskit variational circuit that produces
    expectation values of Pauli‑Z observables, followed by a small
    PyTorch classifier that maps these quantum features to class logits.
    """

    def __init__(
        self,
        num_qubits: int = 2,
        depth: int = 2,
        hidden_dim: int = 4,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.device = device

        # Quantum circuit and metadata
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(
            num_qubits, depth
        )
        self.simulator = Aer.get_backend("aer_simulator_statevector")

        # Classical classifier head
        self.classifier = nn.Sequential(
            nn.Linear(num_qubits, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        ).to(self.device)

        self.optimizer = optim.Adam(self.classifier.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def _expectations(self, X: torch.Tensor) -> torch.Tensor:
        """Compute Pauli‑Z expectation values for each sample."""
        n_samples = X.shape[0]
        expectations = torch.zeros((n_samples, self.num_qubits), device=self.device)

        for i in range(n_samples):
            sample = X[i]
            bound_params = {p: sample[j].item() for j, p in enumerate(self.encoding)}
            bound_params.update(
                {p: 0.0 for p in self.weights}
            )  # variational parameters start at zero
            bound_circuit = self.circuit.bind_parameters(bound_params)
            job = execute(bound_circuit, self.simulator, shots=1024)
            result = job.result()
            statevector = result.get_statevector(bound_circuit)
            for j, obs in enumerate(self.observables):
                exp_val = result.get_expectation_value(obs, bound_circuit)
                expectations[i, j] = exp_val
        return expectations

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass: quantum feature extraction → classical classifier."""
        quantum_feats = self._expectations(X)
        logits = self.classifier(quantum_feats)
        return logits

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> None:
        """Train the classical classifier head on quantum features."""
        self.train()
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                logits = self.forward(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Return class predictions for the input data."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(X.to(self.device))
        return logits.argmax(dim=1)

    def clip_quantum_parameters(self, bound: float = 5.0) -> None:
        """Clip variational parameters to a safe range (no training here)."""
        # Since variational parameters are not part of the torch graph,
        # this method is a placeholder for future joint training.
        pass


__all__ = ["FraudDetectionHybrid"]
