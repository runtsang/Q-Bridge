"""Quantum regression model using Pennylane. Implements an amplitude-encoded
dataset, a multi-layer variational circuit with entangling blocks, and a
classical readout layer. The module is compatible with torch autograd and
supports gradient-based training."""
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from torch.utils.data import Dataset, DataLoader

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate amplitude-encoded states of superposition form and labels."""
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    state_vectors = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        state_vectors[i, 0] = np.cos(thetas[i])
        state_vectors[i, -1] = np.exp(1j * phis[i]) * np.sin(thetas[i])
    labels = np.sin(2 * thetas) * np.cos(phis)
    return state_vectors.astype(np.complex64), labels.astype(np.float32)

class RegressionDataset(Dataset):
    """Wrap amplitude-encoded states and regression targets."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self):  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):
        return {"states": torch.tensor(self.states[idx], dtype=torch.cfloat),
                "target": torch.tensor(self.labels[idx], dtype=torch.float32)}

class QuantumRegressionModel(nn.Module):
    """Hybrid quantum-classical regression model."""
    def __init__(self, num_wires: int, num_layers: int = 3, device: str | torch.device = "cpu"):
        super().__init__()
        self.num_wires = num_wires
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=num_wires, shots=None)
        # Trainable parameters for variational circuit
        self.params = nn.Parameter(torch.randn(num_layers, num_wires, 3))
        self.head = nn.Linear(num_wires, 1)

        # Define QNode
        def circuit(state, params):
            qml.StatePreparation(state, wires=range(num_wires))
            for layer in range(num_layers):
                for w in range(num_wires):
                    qml.RX(params[layer, w, 0], wires=w)
                    qml.RY(params[layer, w, 1], wires=w)
                    qml.RZ(params[layer, w, 2], wires=w)
                for w in range(num_wires - 1):
                    qml.CNOT(wires=[w, w + 1])
            return [qml.expval(qml.PauliZ(w)) for w in range(num_wires)]

        self.qnode = qml.QNode(circuit, self.dev, interface="torch")

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qout = torch.stack([self.qnode(state_batch[i], self.params) for i in range(bsz)])
        return self.head(qout).squeeze(-1)

def train_qmodel(model: nn.Module,
                 dataset: Dataset,
                 epochs: int = 200,
                 batch_size: int = 16,
                 lr: float = 1e-3,
                 clip_norm: float = 1.0,
                 early_stop: int = 20,
                 device: str | torch.device = "cpu") -> tuple[float, float]:
    """Train the quantum model with Adam, gradient clipping, and early stopping."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val = float("inf")
    patience = 0
    history = []
    model.to(device)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            states = batch["states"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad()
            out = model(states)
            loss = criterion(out, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            epoch_loss += loss.item() * states.size(0)
        epoch_loss /= len(dataset)
        if epoch_loss < best_val:
            best_val = epoch_loss
            patience = 0
        else:
            patience += 1
            if patience >= early_stop:
                break
        history.append(epoch_loss)
    return best_val, float(history[-1])

__all__ = ["QuantumRegressionModel", "RegressionDataset", "generate_superposition_data", "train_qmodel"]
