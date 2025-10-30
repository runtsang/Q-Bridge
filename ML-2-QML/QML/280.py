"""AutoencoderQML: a hybrid variational autoencoder built with Pennylane and Qiskit."""

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer
import torch
from torch.utils.data import DataLoader, TensorDataset

class AutoencoderQML:
    """Hybrid variational autoencoder using a parameterised quantum circuit."""
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 3,
                 num_qubits: int = 4,
                 n_layers: int = 2,
                 device: str = "default.qubit",
                 learning_rate: float = 0.01):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_qubits = num_qubits
        self.n_layers = n_layers
        self.device = device
        self.dev = qml.device(device, wires=num_qubits)
        self.qnode = self._build_qnode()
        self.optimizer = AdamOptimizer(learning_rate)

    def _feature_map(self, x):
        """Simple real amplitude feature map."""
        qml.templates.embeddings.RealAmplitudes(x, wires=range(self.num_qubits))

    def _ansatz(self, weights):
        """Parameterized ansatz with alternating layers."""
        for i in range(self.n_layers):
            qml.templates.layers.StronglyEntanglingLayers(weights[i], wires=range(self.num_qubits))

    def _build_qnode(self):
        @qml.qnode(self.dev, interface="torch")
        def circuit(x, weights):
            self._feature_map(x)
            self._ansatz(weights)
            return qml.expval(qml.PauliZ(0))
        return circuit

    def loss(self, batch, weights):
        """Mean‑squared error between target and quantum expectation."""
        preds = []
        for x in batch:
            preds.append(self.qnode(x, weights))
        preds = torch.stack(preds).squeeze()
        target = torch.mean(batch, dim=0)  # simplistic proxy
        return torch.mean((preds - target) ** 2)

    def train(self,
              data: torch.Tensor,
              *,
              epochs: int = 100,
              batch_size: int = 32):
        """Variational training loop."""
        dataset = TensorDataset(_as_tensor(data))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        num_params = self.n_layers * (self.num_qubits * 2)
        weights = torch.randn((self.n_layers, self.num_qubits * 2), requires_grad=True)
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch, in loader:
                batch = batch.to(self.dev)
                loss = self.loss(batch, weights)
                loss.backward()
                weights = self.optimizer.step(weights)
                epoch_loss += loss.item()
            epoch_loss /= len(loader)
            print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f}")
        self.weights = weights

def AutoencoderQMLFactory(input_dim: int,
                          latent_dim: int = 3,
                          num_qubits: int = 4,
                          n_layers: int = 2,
                          device: str = "default.qubit",
                          learning_rate: float = 0.01):
    """Convenience factory returning a ready‑to‑train AutoencoderQML instance."""
    return AutoencoderQML(
        input_dim=input_dim,
        latent_dim=latent_dim,
        num_qubits=num_qubits,
        n_layers=n_layers,
        device=device,
        learning_rate=learning_rate,
    )

__all__ = ["AutoencoderQML", "AutoencoderQMLFactory"]
