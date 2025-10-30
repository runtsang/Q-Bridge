import torch
import torch.nn as nn
import pennylane as qml
import pennylane.numpy as np

class QuantumNatEnhanced(nn.Module):
    """
    Hybrid classical–quantum model.
    Classical CNN extracts features, a linear layer produces
    parameters for a 4‑qubit variational circuit, and the
    circuit outputs 4 expectation values that are linearly
    mapped to 8 logits.
    """
    def __init__(self, num_qubits: int = 4, circuit_depth: int = 2) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.circuit_depth = circuit_depth

        # Classical feature extractor (same as in the pure‑ML version)
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        self._feature_dim = 64 * 3 * 3

        # Map classical features to circuit parameters
        self.param_encoder = nn.Linear(self._feature_dim, num_qubits * circuit_depth * 3)

        # Classical head after the quantum layer
        self.classifier = nn.Linear(num_qubits, 8)
        self.norm = nn.BatchNorm1d(8)

        # Quantum device
        self.dev = qml.device("default.qubit", wires=self.num_qubits, shots=0)

        # Build the variational circuit
        self.circuit = qml.QNode(self._q_circuit, self.dev, interface="torch")

    def _q_circuit(self, params: np.ndarray) -> np.ndarray:
        """
        Variational circuit with alternating layers of
        single‑qubit rotations and CNOT entanglement.
        `params` shape: (depth, num_qubits, 3) for Rx, Ry, Rz.
        """
        depth = self.circuit_depth
        for d in range(depth):
            for q in range(self.num_qubits):
                qml.RX(params[d, q, 0], wires=q)
                qml.RY(params[d, q, 1], wires=q)
                qml.RZ(params[d, q, 2], wires=q)
            # Entangling layer (cyclic CNOTs)
            for q in range(self.num_qubits):
                qml.CNOT(wires=[q, (q + 1) % self.num_qubits])
        # Return expectation values of PauliZ on each wire
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feats = self.features(x)
        flattened = feats.view(bsz, -1)
        # Encode to parameters for the quantum circuit
        params = self.param_encoder(flattened)
        # Reshape to (batch, depth, wires, 3)
        depth = self.circuit_depth
        params = params.view(bsz, depth, self.num_qubits, 3)
        # Run the quantum circuit for each sample in the batch
        q_outputs = self.circuit(params)  # shape (batch, num_qubits)
        logits = self.classifier(q_outputs)
        return self.norm(logits)

__all__ = ["QuantumNatEnhanced"]
