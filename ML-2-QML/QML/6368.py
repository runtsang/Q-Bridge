import pennylane as qml
import pennylane.numpy as np
import torch
import torch.nn as nn

class HybridFunction(nn.Module):
    """Differentiable quantum expectation head using Pennylane qnode."""
    def __init__(self, n_qubits: int = 2, device: str = "default.qubit", shift: float = np.pi/2):
        super().__init__()
        self.shift = shift
        self.n_qubits = n_qubits
        self.dev = qml.device(device, wires=n_qubits)
        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(params: torch.Tensor):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RY(params[i], wires=i)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))
        self.circuit = circuit
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch,)
        probs = self.circuit(x)
        return probs

class QuantumEnhancedHybridNet(nn.Module):
    """CNN backbone followed by a Pennylane quantum expectation head."""
    def __init__(self, n_qubits: int = 2, shift: float = np.pi/2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, n_qubits),
            nn.ReLU(inplace=True),
        )
        self.hybrid = HybridFunction(n_qubits=n_qubits, shift=shift)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        probs = self.hybrid(x).squeeze(-1)
        return torch.stack([probs, 1 - probs], dim=-1)
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    @staticmethod
    def train_step(model: "QuantumEnhancedHybridNet",
                   dataloader: torch.utils.data.DataLoader,
                   optimizer: torch.optim.Optimizer,
                   criterion: nn.Module,
                   device: torch.device) -> float:
        model.train()
        total_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        return total_loss / len(dataloader.dataset)

__all__ = ["HybridFunction", "QuantumEnhancedHybridNet"]
