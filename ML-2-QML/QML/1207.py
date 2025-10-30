import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable, Tuple

class HybridBinaryClassifier(nn.Module):
    """
    Hybrid classical‑quantum binary classifier implemented with Pennylane.
    The quantum part is a variational circuit with entanglement and
    parameter‑shift gradient. The class supports training on a simulator
    or on a real device, early stopping, and exporting the circuit to
    Qiskit.
    """

    def __init__(
        self,
        in_features: int,
        n_qubits: int = 2,
        layers: int = 3,
        device: str = "default.qubit",
        shots: int = 1024,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.n_qubits = n_qubits
        self.layers = layers
        self.shots = shots
        self.dev = qml.device(device, wires=n_qubits, shots=shots)

        # Classical feature extractor
        self.fc1 = nn.Linear(in_features, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Quantum circuit parameters
        self.params = nn.Parameter(torch.randn(n_qubits * layers))

        # Quantum node
        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(inputs: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            # Encode inputs
            for i, w in enumerate(inputs):
                qml.RY(w, wires=i % n_qubits)
            # Variational layers
            for l in range(layers):
                for q in range(n_qubits):
                    qml.RY(params[l * n_qubits + q], wires=q)
                # Entanglement
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        # Quantum expectation
        q_out = self.circuit(x, self.params)
        probs = torch.sigmoid(q_out)
        return torch.cat((probs, 1 - probs), dim=-1)

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    def train_with_early_stopping(
        self,
        train_loader: Iterable,
        val_loader: Iterable,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int = 100,
        patience: int = 10,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        best_val_loss = float("inf")
        epochs_no_improve = 0

        self.to(device)
        for epoch in range(epochs):
            self.train()
            for batch, labels in train_loader:
                batch, labels = batch.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self(batch)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Validation
            val_loss = 0.0
            self.eval()
            with torch.no_grad():
                for batch, labels in val_loader:
                    batch, labels = batch.to(device), labels.to(device)
                    outputs = self(batch)
                    val_loss += criterion(outputs, labels).item()
            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(self.state_dict(), "best_qmodel.pt")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    def export_qiskit(self, file_path: str = "qiskit_circuit.qasm") -> None:
        """
        Export the variational circuit to a Qiskit QASM file.
        """
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(self.n_qubits)
        # Map Pennylane parameters to qiskit gates
        for l in range(self.layers):
            for q in range(self.n_qubits):
                theta = self.params[l * self.n_qubits + q].item()
                qc.ry(theta, q)
            # Entanglement
            for q in range(self.n_qubits - 1):
                qc.cx(q, q + 1)
            qc.cx(self.n_qubits - 1, 0)
        qc.measure_all()
        qc.qasm(filename=file_path)

__all__ = ["HybridBinaryClassifier"]
