import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Iterable, Tuple

import qiskit
from qiskit import Aer
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA

__all__ = ["UnifiedFCLAutoencoder", "AutoencoderConfig", "AutoencoderNet", "QuantumEncoder"]

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """Dense autoencoder network."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        enc_layers = []
        in_dim = config.input_dim
        for h in config.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                enc_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        in_dim = config.latent_dim
        for h in reversed(config.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                dec_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

class QuantumEncoder:
    """Parameterized quantum encoder using RealAmplitudes and a SamplerQNN."""
    def __init__(self, num_qubits: int, backend=None, shots: int = 1024) -> None:
        self.num_qubits = num_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

        self.circuit = qiskit.QuantumCircuit(num_qubits)
        self.input_params = [qiskit.circuit.Parameter(f"x{i}") for i in range(num_qubits)]
        self.ansatz = qiskit.circuit.library.RealAmplitudes(num_qubits, reps=1)
        self.circuit.append(self.ansatz, range(num_qubits))
        self.circuit.measure_all()

        self.weight_params = self.ansatz.parameters

        def interpret(x):
            return x

        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            interpret=interpret,
            output_shape=(num_qubits,),
            sampler=Sampler(),
        )
        init_weights = np.random.randn(len(self.weight_params))
        self.qnn.set_weights(init_weights)

    def run(self, inputs: np.ndarray) -> np.ndarray:
        return self.qnn(inputs)

    def get_weights(self) -> np.ndarray:
        return self.qnn.get_weights()

    def set_weights(self, weights: np.ndarray) -> None:
        self.qnn.set_weights(weights)

class UnifiedFCLAutoencoder(nn.Module):
    """
    Hybrid autoencoder that combines a classical encoder/decoder with a
    parameterized quantum fully‑connected layer.
    """
    def __init__(self,
                 config: AutoencoderConfig,
                 backend=None,
                 shots: int = 1024) -> None:
        super().__init__()
        self.config = config

        autoenc_cfg = AutoencoderConfig(
            input_dim=config.input_dim,
            latent_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
        )
        self.autoencoder_net = AutoencoderNet(autoenc_cfg)
        self.encoder = self.autoencoder_net.encoder
        self.decoder = self.autoencoder_net.decoder

        self.quantum_encoder = QuantumEncoder(
            num_qubits=config.input_dim,
            backend=backend,
            shots=shots,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        latent_np = latent.detach().cpu().numpy()
        q_latent = self.quantum_encoder.run(latent_np)
        q_latent_t = torch.tensor(q_latent, dtype=x.dtype, device=x.device)
        recon = self.decoder(q_latent_t)
        return recon

    def train_classical(self,
                        data: torch.Tensor,
                        epochs: int = 100,
                        batch_size: int = 64,
                        lr: float = 1e-3) -> None:
        dataset = TensorDataset(data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr,
        )
        loss_fn = nn.MSELoss()

        for _ in range(epochs):
            epoch_loss = 0.0
            for batch in loader:
                batch = batch[0].to(next(self.parameters()).device)
                optimizer.zero_grad()
                recon = self.forward(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)

    def train_quantum(self,
                      data: torch.Tensor,
                      epochs: int = 50) -> None:
        weight_shape = len(self.quantum_encoder.get_weights())

        def objective(weights: np.ndarray) -> float:
            self.quantum_encoder.set_weights(weights)

            with torch.no_grad():
                data_np = data.detach().cpu().numpy()
                q_latent = self.quantum_encoder.run(data_np)
                q_latent_t = torch.tensor(q_latent, dtype=data.dtype, device=data.device)
                recon = self.decoder(q_latent_t)
                loss = nn.MSELoss()(recon, data).item()
            return loss

        optimizer = COBYLA(maxiter=2000, disp=False)
        x0 = self.quantum_encoder.get_weights()
        optimizer.optimize(weight_shape, objective, x0)
        self.quantum_encoder.set_weights(optimizer.x)

    def train(self,
              data: torch.Tensor,
              epochs_classical: int = 100,
              epochs_quantum: int = 50) -> None:
        self.train_classical(data, epochs=epochs_classical)
        self.train_quantum(data, epochs=epochs_quantum)
