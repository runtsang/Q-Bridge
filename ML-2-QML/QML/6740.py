import pennylane as qml
import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 4
    n_layers: int = 2
    device: str = "default.qubit"
    shots: int = 1000
    learning_rate: float = 0.01
    max_epochs: int = 200
    batch_size: int = 32

class AutoencoderNet(nn.Module):
    """
    Variational quantum autoencoder built with PennyLane.
    The encoder maps classical data onto a latent quantum state via a feature map
    and a parameterised ansatz. The decoder reconstructs data from the latent
    representation using another parameterised circuit. Training is performed
    with a classical optimiser on the circuit parameters.
    """
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.dev = qml.device(config.device, wires=config.input_dim, shots=config.shots)
        self._build_circuits()

    def _build_circuits(self) -> None:
        @qml.qnode(self.dev, interface="torch")
        def encoder_circuit(inputs, params):
            qml.AngleEmbedding(inputs, wires=range(self.config.input_dim))
            qml.StronglyEntanglingLayers(params, wires=range(self.config.input_dim))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.config.latent_dim)]

        @qml.qnode(self.dev, interface="torch")
        def decoder_circuit(latents, params):
            qml.AngleEmbedding(latents, wires=range(self.config.latent_dim))
            qml.StronglyEntanglingLayers(params, wires=range(self.config.latent_dim))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.config.input_dim)]

        self.encoder_circuit = encoder_circuit
        self.decoder_circuit = decoder_circuit

        # Initialise trainable parameters
        self.encoder_params = nn.Parameter(
            torch.randn(self.config.n_layers, self.config.input_dim, 3)
        )
        self.decoder_params = nn.Parameter(
            torch.randn(self.config.n_layers, self.config.latent_dim, 3)
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder_circuit(inputs, self.encoder_params)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder_circuit(latents, self.decoder_params)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latents = self.encode(inputs)
        return self.decode(latents)

    def compute_loss(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(recon, target, reduction="mean")

    def train_qml(
        self,
        data: torch.Tensor,
        *,
        epochs: int | None = None,
        batch_size: int | None = None,
        lr: float | None = None,
    ) -> list[float]:
        epochs = epochs or self.config.max_epochs
        batch_size = batch_size or self.config.batch_size
        lr = lr or self.config.learning_rate

        optimizer = torch.optim.Adam(
            list(self.parameters()), lr=lr, weight_decay=0
        )

        dataset = torch.utils.data.TensorDataset(data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        history: list[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            self.train()
            for (batch,) in loader:
                batch = batch.to(next(self.parameters()).device)
                optimizer.zero_grad(set_to_none=True)
                recon = self(batch)
                loss = self.compute_loss(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)
        return history

    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> float:
        self.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = batch[0].to(next(self.parameters()).device)
                recon = self(batch)
                loss = self.compute_loss(recon, batch)
                total_loss += loss.item() * batch.size(0)
                total_samples += batch.size(0)
        return total_loss / total_samples

    def save(self, path: str) -> None:
        torch.save(
            {
                "config": self.config,
                "encoder_params": self.encoder_params.detach().clone(),
                "decoder_params": self.decoder_params.detach().clone(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "AutoencoderNet":
        checkpoint = torch.load(path, map_location="cpu")
        config: AutoencoderConfig = checkpoint["config"]
        model = cls(config)
        model.encoder_params.data = checkpoint["encoder_params"]
        model.decoder_params.data = checkpoint["decoder_params"]
        model.eval()
        return model

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 4,
    n_layers: int = 2,
    device: str = "default.qubit",
    shots: int = 1000,
    learning_rate: float = 0.01,
    max_epochs: int = 200,
    batch_size: int = 32,
) -> AutoencoderNet:
    """
    Factory that mirrors the classical helper, returning a quantum autoencoder.
    """
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        n_layers=n_layers,
        device=device,
        shots=shots,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        batch_size=batch_size,
    )
    return AutoencoderNet(config)

__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderNet",
]
