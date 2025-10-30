import pennylane as qml
import numpy as np
import torch
from torch import optim
from dataclasses import dataclass
from typing import Tuple, Iterable, List

__all__ = ["Autoencoder"]

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 3
    encoder_layers: int = 2
    decoder_layers: int = 2
    device: str = "default.qubit"
    shots: int = 1024
    lr: float = 0.01
    epochs: int = 200
    batch_size: int = 32

class Autoencoder:
    """A hybrid quantum autoencoder using Pennylane."""
    def __init__(self, config: AutoencoderConfig) -> None:
        self.config = config
        self.dev = qml.device(config.device, wires=config.input_dim, shots=config.shots)
        # Parameters for encoder and decoder ansatz
        self.encoder_params = torch.nn.Parameter(
            torch.randn(config.encoder_layers * config.input_dim, requires_grad=True)
        )
        self.decoder_params = torch.nn.Parameter(
            torch.randn(config.decoder_layers * config.input_dim, requires_grad=True)
        )
        self.optimizer = optim.Adam(
            [self.encoder_params, self.decoder_params], lr=config.lr
        )
        self._build_qnode()

    def _build_qnode(self) -> None:
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, enc_params: torch.Tensor, dec_params: torch.Tensor) -> torch.Tensor:
            """Return the statevector after encoding and decoding."""
            # Amplitude embedding of the input
            qml.AmplitudeEmbedding(
                features=inputs,
                wires=range(self.config.input_dim),
                normalize=True,
            )
            # Encoder ansatz
            qml.RealAmplitudes(wires=range(self.config.input_dim), reps=self.config.encoder_layers)(enc_params)
            # Decoder ansatz
            qml.RealAmplitudes(wires=range(self.config.input_dim), reps=self.config.decoder_layers)(dec_params)
            return qml.state()
        self.circuit = circuit

    def loss(self, inputs: torch.Tensor) -> torch.Tensor:
        """Mean‑squared error between input and reconstructed statevector."""
        batch_size = inputs.shape[0]
        loss = 0.0
        for i in range(batch_size):
            inp = inputs[i]
            state = self.circuit(inp, self.encoder_params, self.decoder_params)
            loss += torch.sum((state - inp) ** 2)
        return loss / batch_size

    def train(self, data: np.ndarray) -> List[float]:
        """Train the quantum autoencoder."""
        data = np.array(data, dtype=np.float32)
        # Normalize each sample for amplitude encoding
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        data = np.divide(data, norms, out=np.zeros_like(data), where=norms!= 0)
        history: List[float] = []
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            for i in range(0, len(data), self.config.batch_size):
                batch = data[i : i + self.config.batch_size]
                batch_t = torch.tensor(batch, dtype=torch.float32, requires_grad=False)
                self.optimizer.zero_grad()
                loss_val = self.loss(batch_t)
                loss_val.backward()
                self.optimizer.step()
                epoch_loss += loss_val.item() * batch_t.shape[0]
            epoch_loss /= len(data)
            history.append(epoch_loss)
        return history

    def reconstruct(self, input_vec: np.ndarray) -> np.ndarray:
        """Return the reconstructed statevector for a single input."""
        norm = np.linalg.norm(input_vec)
        inp = input_vec / norm if norm > 0 else input_vec
        inp_t = torch.tensor(inp, dtype=torch.float32)
        return self.circuit(inp_t, self.encoder_params, self.decoder_params).detach().numpy()
