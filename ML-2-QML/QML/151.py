"""Quantum regression model leveraging PennyLane."""
import numpy as np
import torch
import pennylane as qml
from pennylane import numpy as pnp
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate amplitude‑encoded superposition states and target values."""
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i, 0] = np.cos(thetas[i])
        states[i, -1] = np.exp(1j * phis[i]) * np.sin(thetas[i])
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset for quantum regression tasks."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class RegressionModel:
    """Variational quantum circuit for regression with a classical head."""
    def __init__(self, num_wires: int, dev: qml.Device | None = None):
        self.num_wires = num_wires
        self.dev = dev or qml.device("default.qubit", wires=num_wires, shots=200)
        self.params_shape = (num_wires, 3)  # one layer of RX,RZ,RX per wire
        self.params = pnp.random.randn(*self.params_shape)
        self.head = nn.Linear(num_wires, 1)
        self.circuit = qml.QNode(self._circuit_impl, dev=self.dev)

    # ------------------------------------------------------------------
    # Quantum circuit
    # ------------------------------------------------------------------
    def _circuit_impl(self, x: np.ndarray, params: np.ndarray):
        """Amplitude‑encoded input followed by a parameterised ansatz."""
        # State preparation
        qml.QubitStateVector(x, wires=range(self.num_wires))
        # Entangling layer
        for i in range(self.num_wires):
            qml.CNOT(wires=[i, (i + 1) % self.num_wires])
        # Parameterised rotation layer
        for i in range(self.num_wires):
            qml.RX(params[i, 0], wires=i)
            qml.RZ(params[i, 1], wires=i)
            qml.RX(params[i, 2], wires=i)
        # Expectation values of PauliZ on each wire
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_wires)]

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass: quantum feature extraction + classical head."""
        # Convert to numpy for PennyLane
        x_np = batch.cpu().numpy()
        # Evaluate circuit for each sample in batch
        features = np.array([self.circuit(x, self.params) for x in x_np])
        # Convert to torch tensor
        features_t = torch.tensor(features, dtype=torch.float32, device=batch.device)
        return self.head(features_t).squeeze(-1)

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 200,
        lr: float = 0.01,
        device: str = "cpu",
        patience: int | None = 20,
    ) -> None:
        """Train the hybrid model using Adam on the classical parameters."""
        self.head.to(device)
        optimizer = torch.optim.Adam(list(self.head.parameters()), lr=lr)
        criterion = nn.MSELoss()
        best_val = float("inf")
        no_improve = 0

        for epoch in range(1, epochs + 1):
            self.head.train()
            epoch_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                preds = self.forward(batch["states"].to(device))
                loss = criterion(preds, batch["target"].to(device))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch["states"].size(0)
            epoch_loss /= len(train_loader.dataset)

            if val_loader is not None:
                val_loss = self.evaluate(val_loader, device=device)
                if val_loss < best_val:
                    best_val = val_loss
                    no_improve = 0
                    torch.save(self.head.state_dict(), "_best_qmodel.pt")
                else:
                    no_improve += 1
                if patience is not None and no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d} | Train loss: {epoch_loss:.4f}" + (f" | Val loss: {val_loss:.4f}" if val_loader else ""))

        if val_loader is not None:
            self.head.load_state_dict(torch.load("_best_qmodel.pt"))

    def predict(self, loader: DataLoader, device: str = "cpu") -> torch.Tensor:
        """Return predictions for a dataset."""
        self.head.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                preds.append(self.forward(batch["states"].to(device)).cpu())
        return torch.cat(preds)

    def evaluate(self, loader: DataLoader, device: str = "cpu") -> float:
        """Compute mean squared error on a dataset."""
        preds = self.predict(loader, device=device)
        targets = torch.cat([batch["target"] for batch in loader])
        return nn.functional.mse_loss(preds, targets).item()


__all__ = ["RegressionModel", "RegressionDataset", "generate_superposition_data"]
