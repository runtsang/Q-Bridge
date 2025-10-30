import pennylane as qml
import torch
import torch.nn as nn

class QuantumNATHybrid(nn.Module):
    """
    Hybrid model: CNN encoder + 4‑qubit variational circuit + classical MLP head.
    Uses PennyLane for the quantum part with a torch interface.
    """

    def __init__(self, num_classes: int = 4, device: str = "cpu"):
        super().__init__()
        # Feature extractor: CNN
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Linear mapping from CNN features to 4‑dimensional input for the circuit
        self.input_mapping = nn.Linear(16 * 7 * 7, 4)
        # Quantum device and circuit
        self.n_wires = 4
        self.dev = qml.device("default.qubit", wires=self.n_wires, shots=1024)
        def quantum_circuit(x, weights):
            # Encode the 4‑dimensional input
            for i in range(self.n_wires):
                qml.RY(x[i], wires=i)
            # Variational layer
            for i in range(self.n_wires):
                qml.RZ(weights[i], wires=i)
            # Entangling layer
            for i in range(self.n_wires - 1):
                qml.CNOT(wires=[i, i + 1])
            # Return expectation values of PauliZ on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]
        self.quantum_circuit = qml.QNode(quantum_circuit, self.dev, interface="torch")
        # Trainable weights for the variational layer
        self.weights = nn.Parameter(torch.randn(self.n_wires))
        # Classifier head
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.n_wires),
            nn.Linear(self.n_wires, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.feature_extractor(x)
        flattened = features.view(bsz, -1)
        # Map to 4‑dimensional vector for the circuit
        x_map = self.input_mapping(flattened)  # shape (bsz, 4)
        # Compute quantum circuit outputs for each sample
        # Using torch.vmap for vectorized execution if available
        try:
            q_out = torch.vmap(self.quantum_circuit, in_dims=(0, None))(x_map, self.weights)
        except AttributeError:
            # Fallback to explicit loop for older PyTorch versions
            q_out = torch.stack([self.quantum_circuit(x_map[i], self.weights) for i in range(bsz)], dim=0)
        logits = self.classifier(q_out)
        return logits
