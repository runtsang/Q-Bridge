import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN

def QuantumSamplerQNN() -> QiskitSamplerQNN:
    inputs = ParameterVector("input", 2)
    weights = ParameterVector("weight", 4)
    qc = QuantumCircuit(2)
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[2], 0)
    qc.ry(weights[3], 1)
    sampler = StatevectorSampler()
    return QiskitSamplerQNN(circuit=qc, input_params=inputs, weight_params=weights, sampler=sampler)

class QFCModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 4))
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)

class HybridAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 3, num_trash: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        self.encoder_circuit = self._build_encoder()
        self.decoder_circuit = self._build_decoder()
        self.sampler = StatevectorSampler()
        self.classifier = QFCModel()

    def _build_encoder(self):
        num_qubits = self.latent_dim + 2 * self.num_trash + 1
        qr = QuantumRegister(num_qubits, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.compose(RealAmplitudes(num_qubits, reps=5), range(0, num_qubits), inplace=True)
        return circuit

    def _build_decoder(self):
        num_qubits = self.latent_dim + 2 * self.num_trash + 1
        qr = QuantumRegister(num_qubits, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.h(num_qubits - 1)
        for i in range(self.num_trash):
            circuit.cswap(num_qubits - 1, self.latent_dim + i, self.latent_dim + self.num_trash + i)
        circuit.h(num_qubits - 1)
        circuit.measure(num_qubits - 1, cr[0])
        return circuit

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Simplified placeholder: project input onto latent dimension
        return x[:, :self.latent_dim]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Simplified placeholder: reconstruct to input dimension
        pad = torch.zeros(z.shape[0], self.input_dim - self.latent_dim, device=z.device)
        return torch.cat([z, pad], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        recon = self.decode(z)
        return recon

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x.unsqueeze(1))

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 3,
    num_trash: int = 2,
) -> HybridAutoencoder:
    return HybridAutoencoder(input_dim, latent_dim, num_trash)

def train_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> list[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

def _as_tensor(data: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)

__all__ = [
    "Autoencoder",
    "HybridAutoencoder",
    "train_autoencoder",
    "QuantumSamplerQNN",
    "QFCModel",
]
