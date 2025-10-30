import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from dataclasses import dataclass
from typing import Tuple

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class HybridQuantumAutoencoder(tq.QuantumModule):
    """Hybrid quantum‑classical autoencoder.  The CNN and classical MLPs are the same
    as in the classical variant; the latent representation is encoded into a small
    quantum device, processed by a variational layer, and measured to produce a
    new latent vector that is fed to the classical decoder."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=2)
            self.crx(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=0)
            tqf.sx(qdev, wires=1)
            tqf.cnot(qdev, wires=[2, 0])

    def __init__(self, config: AutoencoderConfig, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        # CNN encoder (identical to classical)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_proj = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, config.latent_dim),
        )
        # Classical encoder MLP
        encoder_layers = []
        in_dim = config.latent_dim
        for hidden in config.hidden_dims:
            encoder_layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.mlp_encoder = nn.Sequential(*encoder_layers)
        # Quantum latent layer
        self.encoder_q = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical decoder MLP
        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.mlp_decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the quantum‑processed latent vector."""
        bsz = x.shape[0]
        features = self.cnn(x)
        flat = features.view(bsz, -1)
        latent = self.fc_proj(flat)
        latent = self.mlp_encoder(latent)
        # Quantum embedding
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        self.encoder_q(qdev, latent)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return out

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp_decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

__all__ = ["HybridQuantumAutoencoder", "AutoencoderConfig"]
