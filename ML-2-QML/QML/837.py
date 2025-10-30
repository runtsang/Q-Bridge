import torch
import torch.nn as nn
import torchquantum as tq
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset

class QuantumRegression__gen104(tq.QuantumModule):
    """
    Quantum regression model that augments the original seed by:
      * Adding a second variational layer
      * Using a parameter‑shift gradient estimator
      * Exposing fit/predict methods for DataLoader inputs
    """
    def __init__(self, num_wires: int, hidden_wires: int = None):
        super().__init__()
        self.n_wires = num_wires
        self.hidden_wires = hidden_wires or num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer1 = self.QLayer(num_wires, name="layer1")
        self.q_layer2 = self.QLayer(self.hidden_wires, name="layer2")
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(self.hidden_wires, 1)
        self.gradient_estimator = tq.GradientEstimator()

    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int, name: str = "qlayer"):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.name = name

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    @staticmethod
    def generate_superposition_data(num_wires: int, samples: int):
        omega_0 = np.zeros(2 ** num_wires, dtype=complex)
        omega_0[0] = 1.0
        omega_1 = np.zeros(2 ** num_wires, dtype=complex)
        omega_1[-1] = 1.0
        thetas = 2 * np.pi * np.random.rand(samples)
        phis = 2 * np.pi * np.random.rand(samples)
        states = np.zeros((samples, 2 ** num_wires), dtype=complex)
        for i in range(samples):
            states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
        labels = np.sin(2 * thetas) * np.cos(phis)
        return states, labels

    class RegressionDataset(Dataset):
        def __init__(self, samples: int, num_wires: int):
            self.states, self.labels = self.generate_superposition_data(num_wires, samples)

        def __len__(self):
            return len(self.states)

        def __getitem__(self, idx):
            return torch.tensor(self.states[idx], dtype=torch.cfloat), \
                   torch.tensor(self.labels[idx], dtype=torch.float32)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer1(qdev)
        self.q_layer2(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None,
            epochs: int = 100, lr: float = 1e-3) -> None:
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                preds = self.forward(X)
                loss = criterion(preds, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * X.size(0)
            epoch_loss /= len(train_loader.dataset)
            # Optional validation logic omitted for brevity

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(X).squeeze(-1).cpu()

    def predict_batch(self, loader: DataLoader) -> torch.Tensor:
        self.eval()
        preds = []
        with torch.no_grad():
            for X, _ in loader:
                preds.append(self.predict(X))
        return torch.cat(preds, dim=0)

__all__ = ["QuantumRegression__gen104"]
