import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchquantum as tq
from Autoencoder import Autoencoder
from GraphQNN import fidelity_adjacency

class RegressionDataset(Dataset):
    """Dataset generating superposition‑based regression targets for quantum input."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = self._generate_superposition_data(num_wires, samples)

    @staticmethod
    def _generate_superposition_data(num_wires: int, samples: int):
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
        return states, labels.astype(np.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class HybridRegressionModel(tq.QuantumModule):
    """Hybrid quantum regression model using a torchquantum encoder, parameterized layer, and a classical head."""
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(
        self,
        num_wires: int,
        latent_dim: int = 32,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        graph_threshold: float = 0.9,
        graph_secondary: float | None = None,
    ):
        super().__init__()
        self.num_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Sequential(
            nn.Linear(num_wires, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.autoencoder = Autoencoder(
            input_dim=2 ** num_wires,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.latent_to_angles = nn.Linear(latent_dim, num_wires)
        self.graph_threshold = graph_threshold
        self.graph_secondary = graph_secondary

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        latent = self.autoencoder.encode(state_batch)
        angles = self.latent_to_angles(latent)
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, angles)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

    def compute_graph(self, latent: torch.Tensor) -> torch.Tensor:
        states = [latent[i] for i in range(latent.size(0))]
        graph = fidelity_adjacency(states, self.graph_threshold, secondary=self.graph_secondary)
        adjacency = torch.zeros((latent.size(0), latent.size(0)), dtype=torch.float32)
        for i, j, data in graph.edges(data=True):
            adjacency[i, j] = data.get("weight", 1.0)
        return adjacency

    def fit(self, dataset: Dataset, epochs: int = 20, lr: float = 1e-3, device: torch.device | None = None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            for batch in loader:
                states = batch["states"].to(device)
                target = batch["target"].to(device)
                optimizer.zero_grad()
                pred = self(states)
                loss = loss_fn(pred, target)
                loss.backward()
                optimizer.step()

    def predict(self, states: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self(states)

__all__ = ["HybridRegressionModel", "RegressionDataset"]
