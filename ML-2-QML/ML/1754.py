import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable, List

import sympy
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        return data.to(dtype=torch.float32)
    return torch.as_tensor(data, dtype=torch.float32)


@dataclass
class AutoencoderConfig:
    """Configuration values for :class:`AutoencoderNet`."""
    input_dim: int
    latent_dim: int = 4
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.0
    lambda_fidelity: float = 0.1  # weight of the quantum fidelity term


class AutoencoderNet(nn.Module):
    """
    A hybrid classical–quantum autoencoder.
    The classical encoder/decoder are standard MLPs.
    A differentiable quantum circuit (SamplerQNN) is used to compute a fidelity
    loss between the latent vector and a fixed reference state.
    """
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # --- Classical encoder ------------------------------------------------
        encoder_layers: List[nn.Module] = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # --- Classical decoder ------------------------------------------------
        decoder_layers: List[nn.Module] = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # --- Quantum fidelity circuit -----------------------------------------
        # The latent vector is fed into a RealAmplitudes circuit.
        # The circuit acts as a feature map; the output state is compared
        # to the |0...0> reference state.
        input_symbols = [sympy.Symbol(f"x{i}") for i in range(cfg.latent_dim)]
        quantum_circ = RealAmplitudes(num_qubits=cfg.latent_dim, reps=1)
        self.qnn = SamplerQNN(
            circuit=quantum_circ,
            input_params=input_symbols,
            weight_params=[],
            interpret=lambda x: x,          # identity
            output_shape=2 ** cfg.latent_dim,
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

    def fidelity_loss(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Compute a differentiable fidelity loss.
        The quantum circuit maps the latent vector to a statevector;
        fidelity with |0...0> is |psi[0]|^2.
        """
        qnn_output = self.qnn(latents)                     # shape: (batch, 2**n)
        fidelity = torch.abs(qnn_output[:, 0]) ** 2        # first amplitude
        return torch.mean(1.0 - fidelity)                  # 1 - fidelity


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 4,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.0,
    lambda_fidelity: float = 0.1,
) -> AutoencoderNet:
    """Factory that returns a configured AutoencoderNet."""
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        lambda_fidelity=lambda_fidelity,
    )
    return AutoencoderNet(cfg)


def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """
    Simple reconstruction training loop returning the loss history.
    The loss is a weighted sum of MSE reconstruction loss and quantum fidelity loss.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            latent = model.encode(batch)
            recon = model.decode(latent)
            recon_loss = loss_fn(recon, batch)

            fidelity_loss = model.fidelity_loss(latent)
            loss = recon_loss + model.cfg.lambda_fidelity * fidelity_loss

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)

        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = ["Autoencoder", "AutoencoderConfig", "AutoencoderNet", "train_autoencoder"]
