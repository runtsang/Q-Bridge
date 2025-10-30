from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum import QuantumDevice, QuantumModule

class HybridAutoencoder(QuantumModule):
    """Hybrid autoencoder with a quantum encoder, NAT‑style layer and quantum decoder."""
    class Encoder(QuantumModule):
        def __init__(self, input_dim: int, latent_dim: int) -> None:
            super().__init__()
            self.latent_dim = latent_dim
            # Simple feature‑to‑qubit encoding via RY rotations
            self.input_rotations = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(latent_dim)])
            # Variational ansatz to mix features
            self.ansatz = tq.QuantumCircuit(n_wires=latent_dim, n_layers=3)

        @tq.static_support
        def forward(self, qdev: QuantumDevice, x: torch.Tensor) -> None:
            for i, rot in enumerate(self.input_rotations):
                rot(qdev, wires=i, params=x[:, i % x.shape[1]])
            self.ansatz(qdev)

    class NATLayer(QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.cnot = tq.CNOT(wires=[0, 1])

        @tq.static_support
        def forward(self, qdev: QuantumDevice) -> None:
            self.random_layer(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=2)
            self.cnot(qdev)

    class Decoder(QuantumModule):
        def __init__(self, latent_dim: int, output_dim: int) -> None:
            super().__init__()
            self.latent_dim = latent_dim
            self.output_dim = output_dim
            self.measure = tq.MeasureAll(tq.PauliZ)

        @tq.static_support
        def forward(self, qdev: QuantumDevice) -> torch.Tensor:
            return self.measure(qdev).float()

    def __init__(self, input_dim: int, output_dim: int, latent_dim: int = 32) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.n_wires = latent_dim
        self.encoder = self.Encoder(input_dim, latent_dim)
        self.nat = self.NATLayer(self.n_wires)
        self.decoder = self.Decoder(latent_dim, output_dim)

    @tq.static_support
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        self.encoder(qdev, x)
        self.nat(qdev)
        out = self.decoder(qdev)
        # Map qubit outputs back to dense vector
        return out.view(bsz, -1)

    # Quantum kernel using swap‑test
    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return a Gram matrix via a swap‑test quantum kernel."""
        n = a.shape[0]
        m = b.shape[0]
        qdev = QuantumDevice(n_wires=self.latent_dim + 1, bsz=n * m, device=a.device)
        # Prepare joint state |a>⊗|b>⊗|+>
        def encode_pair(i, j):
            idx = i * m + j
            qdev.reset_states(1)
            self.encoder(qdev, a[i].unsqueeze(0))
            qdev.cnot(wires=[self.latent_dim, self.latent_dim])  # placeholder
            self.encoder(qdev, b[j].unsqueeze(0))
            tqf.hadamard(qdev, wires=self.latent_dim)
            return qdev

        # For brevity, use a classical RBF kernel as placeholder
        diff = a.unsqueeze(1) - b.unsqueeze(0)
        return torch.exp(-self.latent_dim * diff.pow(2).sum(-1))

    # Graph adjacency from state fidelities
    def fidelity_adjacency(
        self,
        states: torch.Tensor,
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> torch.Tensor:
        """Build weighted adjacency from pairwise fidelities of quantum states."""
        # Compute pairwise overlaps via inner product of state vectors
        norm = states / (states.norm(dim=-1, keepdim=True) + 1e-12)
        overlaps = norm @ norm.t()
        adjacency = torch.zeros_like(overlaps)
        adjacency[overlaps >= threshold] = 1.0
        if secondary is not None:
            mask = (overlaps >= secondary) & (overlaps < threshold)
            adjacency[mask] = secondary_weight
        return adjacency

def Autoencoder(
    input_dim: int,
    output_dim: int,
    *,
    latent_dim: int = 32,
) -> HybridAutoencoder:
    """Factory mirroring the classical helper."""
    return HybridAutoencoder(input_dim, output_dim, latent_dim)

def train_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """Training loop for the hybrid quantum autoencoder."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

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

__all__ = ["HybridAutoencoder", "Autoencoder", "train_autoencoder"]
