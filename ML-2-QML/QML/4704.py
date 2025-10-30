import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple

# --------------------------------------------------------------------------- #
#  Classical‑looking CNN feature extractor
# --------------------------------------------------------------------------- #
class ConvEncoder(nn.Module):
    """CNN that projects an image to a `feat_dim` vector."""
    def __init__(self, feat_dim: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.project = nn.Linear(16 * 7 * 7, feat_dim)
        self.norm = nn.BatchNorm1d(feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        flat = feats.view(x.shape[0], -1)
        return self.norm(self.project(flat))

# --------------------------------------------------------------------------- #
#  Quantum auto‑encoder
# --------------------------------------------------------------------------- #
class QuantumAutoencoder(nn.Module):
    """Encodes a batch of feature vectors into a lower‑dimensional qubit space
    using a random layer and angle encoding. The first `latent_dim` measurement
    results are returned as a classical latent vector."""
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(input_dim)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.input_dim, bsz=bsz, device=x.device)
        # Encode each feature as an angle on a dedicated qubit
        for i in range(self.input_dim):
            tqf.rx(qdev, x[:, i], wires=i)
        self.random_layer(qdev)
        out = self.measure(qdev)  # shape (bsz, input_dim)
        return out[:, :self.latent_dim]

# --------------------------------------------------------------------------- #
#  Quantum LSTM cell
# --------------------------------------------------------------------------- #
class QuantumQLSTM(nn.Module):
    """LSTM where each gate is a small quantum circuit."""
    class QGate(nn.Module):
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
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            # Encode the classical vector onto the qubits
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                tgt = wire + 1 if wire + 1 < self.n_wires else 0
                tqf.cnot(qdev, wires=[wire, tgt])
            out = self.measure(qdev)
            # Return the first qubit as a scalar gate activation
            return out[:, 0]

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.forget_gate = self.QGate(n_qubits)
        self.input_gate = self.QGate(n_qubits)
        self.update_gate = self.QGate(n_qubits)
        self.output_gate = self.QGate(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
            g = torch.tanh(self.update_gate(self.linear_update(combined)))
            o = torch.sigmoid(self.output_gate(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

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

# --------------------------------------------------------------------------- #
#  Unified model
# --------------------------------------------------------------------------- #
class UnifiedQuantumAutoLSTM(nn.Module):
    """End‑to‑end quantum model that processes image sequences."""
    def __init__(
        self,
        conv_feat_dim: int = 64,
        ae_latent_dim: int = 32,
        lstm_hidden_dim: int = 64,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = ConvEncoder(conv_feat_dim)
        self.autoencoder = QuantumAutoencoder(conv_feat_dim, ae_latent_dim)
        self.lstm = QuantumQLSTM(ae_latent_dim, lstm_hidden_dim, n_qubits=ae_latent_dim)
        self.classifier = nn.Linear(lstm_hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (batch, seq_len, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits over `num_classes`.
        """
        B, seq_len, C, H, W = x.shape
        feats = self.encoder(x.view(B * seq_len, C, H, W))  # (B*seq_len, feat_dim)
        latents = self.autoencoder.encode(feats)  # (B*seq_len, latent_dim)
        latents = latents.view(B, seq_len, -1)
        lstm_out, _ = self.lstm(latents.permute(1, 0, 2))  # (seq_len, batch, hidden)
        hidden = lstm_out[-1]  # last hidden state
        logits = self.classifier(hidden)
        return F.log_softmax(logits, dim=1)

__all__ = ["UnifiedQuantumAutoLSTM"]
