from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Tuple, Optional

# Import the quantum encoder defined in the accompanying QML module
try:
    from. import qml as qml_module
    QuantumAutoencoder = qml_module.HybridAutoencoder
except Exception:
    # Fallback placeholder if the quantum module is not available
    class QuantumAutoencoder(nn.Module):
        def __init__(self, *_, **__):  # pragma: no cover
            super().__init__()
            raise ImportError("Quantum module not available")

# Optional quantum LSTM implementation (copied from reference pair 3)
import torchquantum as tq
import torchquantum.functional as tqf

class QLSTM(nn.Module):
    """Quantum LSTM cell using small quantum circuits for each gate."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                    {"input_idx": [2], "func": "rx", "wires": [2]},
                    {"input_idx": [3], "func": "rx", "wires": [3]},
                ]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                if wire == self.n_wires - 1:
                    tqf.cnot(qdev, wires=[wire, 0])
                else:
                    tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

@dataclass
class AutoencoderConfig:
    """Configuration for the hybrid autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    use_cnn_encoder: bool = False
    use_lstm_decoder: bool = False
    use_q_lstm_decoder: bool = False

class HybridAutoencoder(nn.Module):
    """Hybrid autoencoder that can combine a CNN encoder, a quantum encoder, and a (quantum) LSTM decoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        # Encoder
        if config.use_cnn_encoder:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            # After two 2x2 pools on 28x28 images -> 7x7 feature map
            encoder_output_dim = 16 * 7 * 7
        else:
            self.encoder = nn.Sequential(
                nn.Linear(config.input_dim, config.hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            )
            encoder_output_dim = config.hidden_dims[1]

        # Quantum encoder
        self.quantum = QuantumAutoencoder(input_dim=encoder_output_dim, latent_dim=config.latent_dim)

        # Decoder
        if config.use_lstm_decoder:
            if config.use_q_lstm_decoder:
                self.decoder = QLSTM(input_dim=config.latent_dim, hidden_dim=config.hidden_dims[1], n_qubits=config.hidden_dims[1])
                self.output_layer = nn.Linear(config.hidden_dims[1], config.input_dim)
            else:
                self.decoder = nn.LSTM(config.latent_dim, config.hidden_dims[1], batch_first=True)
                self.output_layer = nn.Linear(config.hidden_dims[1], config.input_dim)
        else:
            self.decoder = nn.Sequential(
                nn.Linear(config.latent_dim, config.hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dims[1], config.input_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        if isinstance(self.encoder, nn.Sequential) and isinstance(self.encoder[0], nn.Conv2d):
            enc = self.encoder(x)
            enc = enc.view(x.size(0), -1)
        else:
            enc = self.encoder(x)

        # Quantum encoding
        latent = self.quantum(enc)

        # Decoder
        if isinstance(self.decoder, nn.Sequential):
            recon = self.decoder(latent)
        else:
            # LSTM path
            latent_seq = latent.unsqueeze(1)  # (batch, seq_len=1, latent_dim)
            dec_out, _ = self.decoder(latent_seq)
            recon = self.output_layer(dec_out.squeeze(1))
        return recon

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    use_cnn_encoder: bool = False,
    use_lstm_decoder: bool = False,
    use_q_lstm_decoder: bool = False,
) -> HybridAutoencoder:
    """Factory that returns a configured hybrid autoencoder."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_cnn_encoder=use_cnn_encoder,
        use_lstm_decoder=use_lstm_decoder,
        use_q_lstm_decoder=use_q_lstm_decoder,
    )
    return HybridAutoencoder(config)

def train_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Training loop for the hybrid autoencoder."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "HybridAutoencoder",
    "train_autoencoder",
]
