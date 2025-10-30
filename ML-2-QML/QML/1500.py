import pennylane as qml
import torch
import torch.nn as nn

class QuantumNATExtendedQML(nn.Module):
    """
    Quantum‑enhanced counterpart of QuantumNATExtended.
    Encodes a 4‑dimensional feature vector into a 4‑qubit state, processes it
    with a parameter‑dependent variational circuit, and decodes the expectation
    values to a 4‑class probability vector.
    """

    def __init__(self, num_classes: int = 4, wires: int = 4, n_layers: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.wires = wires
        self.n_layers = n_layers

        # Pennylane device
        self.dev = qml.device("default.qubit", wires=self.wires, shots=1024)

        # Parameter tensors for the variational circuit
        self.params = nn.Parameter(torch.randn(self.n_layers * self.wires * 3))

        # Classical encoder: compress image to 4‑dimensional vector
        self.encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(16 * 6 * 6, 4),
        )

        # Classical post‑processing head
        self.fc = nn.Linear(self.wires, num_classes)
        self.norm = nn.BatchNorm1d(num_classes)

    def q_circuit(self, x, params):
        """Variational circuit with alternating RX, RY, RZ rotations and CNOT entanglers."""
        params = params.view(self.n_layers, self.wires, 3)
        for layer, (rx, ry, rz) in enumerate(params):
            for w in range(self.wires):
                qml.RX(rx, wires=w)
                qml.RY(ry, wires=w)
                qml.RZ(rz, wires=w)
            # Entangling CNOT chain
            for w in range(self.wires - 1):
                qml.CNOT(wires=[w, w + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.wires)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode classical data
        encoded = self.encoder(x)
        # Apply quantum circuit
        q_out = qml.QNode(self.q_circuit, self.dev)(encoded, self.params)
        q_out = torch.tensor(q_out, dtype=torch.float32, device=x.device)
        # Classical post‑processing
        logits = self.fc(q_out)
        return self.norm(logits)

__all__ = ["QuantumNATExtendedQML"]
