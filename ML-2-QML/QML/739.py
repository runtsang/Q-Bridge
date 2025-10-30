"""Quantum regression model using Pennylane variational circuit."""
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from torch.utils.data import Dataset, DataLoader

def generate_quantum_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate quantum states |ψ(θ,φ)> = cosθ|0...0> + e^{iφ} sinθ|1...1> and target."""
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        omega0 = np.zeros(2 ** num_wires, dtype=complex)
        omega0[0] = 1.0
        omega1 = np.zeros(2 ** num_wires, dtype=complex)
        omega1[-1] = 1.0
        states[i] = np.cos(thetas[i]) * omega0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_quantum_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return torch.tensor(self.states[idx], dtype=torch.cfloat), torch.tensor(self.labels[idx], dtype=torch.float32)

class QModel(nn.Module):
    """Pennylane variational circuit with entanglement and classical head."""
    def __init__(self, num_wires: int, num_layers: int = 3, device: str = "default.qubit"):
        super().__init__()
        self.num_wires = num_wires
        self.num_layers = num_layers
        self.dev = qml.device(device, wires=num_wires)

        def circuit(state, params):
            qml.StatePreparation(state, wires=range(num_wires))
            for layer in range(num_layers):
                for wire in range(num_wires):
                    qml.RX(params[layer, wire, 0], wires=wire)
                    qml.RY(params[layer, wire, 1], wires=wire)
                for wire in range(num_wires - 1):
                    qml.CNOT(wires=[wire, wire + 1])
                qml.CNOT(wires=[num_wires - 1, 0])
            return [qml.expval(qml.PauliZ(w)) for w in range(num_wires)]

        self.qnode = qml.QNode(circuit, self.dev, interface="torch")
        self.params = nn.Parameter(torch.randn(num_layers, num_wires, 2))
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        features = torch.stack([self.qnode(state) for state in state_batch], dim=0)
        return self.head(features).squeeze(-1)

    def fit(self, train_loader: DataLoader, epochs: int = 20, lr: float = 1e-3, device: str | torch.device = "cpu") -> None:
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_state, batch_y in train_loader:
                batch_state, batch_y = batch_state.to(device), batch_y.to(device)
                optimizer.zero_grad()
                preds = self(batch_state)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_state.size(0)
            epoch_loss /= len(train_loader.dataset)
            print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f}")

    def predict(self, X: torch.Tensor, device: str | torch.device = "cpu") -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self(X.to(device)).cpu()

__all__ = ["QModel", "RegressionDataset", "generate_quantum_data"]
