import pennylane as qml
import torch

class QuantumClassifier:
    """
    Quantum variational classifier that mirrors the classical interface.
    Uses PennyLane for parameterised ansatz and automatic gradients.
    """

    def __init__(self, num_qubits: int, depth: int = 3, device: str = "default.qubit"):
        self.num_qubits = num_qubits
        self.depth = depth
        self.dev = qml.device(device, wires=num_qubits)
        # Initialise parameters
        self.weights = torch.nn.Parameter(torch.randn(num_qubits * depth))
        self.encoding = torch.nn.Parameter(torch.randn(num_qubits))
        self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="torch")
        def circuit(x, weights, encoding):
            # Data encoding
            for i in range(self.num_qubits):
                qml.RX(x[i] + encoding[i], i)
            # Variational layers
            idx = 0
            for _ in range(self.depth):
                for i in range(self.num_qubits):
                    qml.RY(weights[idx], i)
                    idx += 1
                for i in range(self.num_qubits - 1):
                    qml.CZ(i, i + 1)
            # Measurements: use first two qubits to represent logits
            return [qml.expval(qml.PauliZ(i)) for i in range(2)]
        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is 1‑D vector of length num_qubits
        return self.circuit(x, self.weights, self.encoding)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(x)
        logits = torch.stack(outputs, dim=1)
        return torch.argmax(logits, dim=1)
