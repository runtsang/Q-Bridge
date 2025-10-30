"""Quantum‑enhanced hybrid model combining quanvolution, QCNN, autoencoder, and optional LSTM."""

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple

# --- Auxiliary quantum modules -------------------------------------------
class QuantumEncoder(tq.QuantumModule):
    """Variational encoder that maps a classical vector to a quantum state."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.var_layer = tq.RandomLayer(n_ops=8, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, n_wires)
        qdev = tq.QuantumDevice(self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        self.var_layer(qdev)
        return self.measure(qdev)

class QLSTMQuantum(tq.QuantumModule):
    """Quantum LSTM cell where each gate is a small variational circuit."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Quantum gate modules
        self.forget_gate = self._make_gate()
        self.input_gate  = self._make_gate()
        self.update_gate = self._make_gate()
        self.output_gate = self._make_gate()

        # Classical linear projections
        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_lin  = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits)

    def _make_gate(self) -> tq.QuantumModule:
        class Gate(tq.QuantumModule):
            def __init__(self, n_wires: int):
                super().__init__()
                self.n_wires = n_wires
                self.encoder = tq.GeneralEncoder(
                    [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
                )
                self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
                self.encoder(qdev, x)
                for gate in self.params:
                    gate(qdev)
                return self.measure(qdev)
        return Gate(self.n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.forget_lin(combined)))
            i = torch.sigmoid(self.input_gate(self.input_lin(combined)))
            g = torch.tanh(self.update_gate(self.update_lin(combined)))
            o = torch.sigmoid(self.output_gate(self.output_lin(combined)))
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
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

# --- Main hybrid model -----------------------------------------------
class QuanvolutionHybrid(tq.QuantumModule):
    """Quantum‑enhanced hybrid model mirroring the classical counterpart."""
    def __init__(
        self,
        num_classes: int = 10,
        use_lstm: bool = False,
        lstm_hidden: int = 64,
        seq_len: int = 1,
        quanv_qubits: int = 4,
        qcnn_qubits: int = 4,
        auto_latent: int = 4,
        lstm_qubits: int = 4,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.use_lstm = use_lstm
        self.seq_len = seq_len

        # 1. Quantum quanvolution filter
        self.qfilter = QuantumEncoder(quanv_qubits)

        # 2. QCNN‑style quantum block
        self.qcnn = QuantumEncoder(qcnn_qubits)

        # 3. Quantum autoencoder (encoder only)
        self.autoencoder = QuantumEncoder(auto_latent)

        # 4. Optional quantum LSTM
        if use_lstm:
            self.lstm = QLSTMQuantum(input_dim=auto_latent,
                                     hidden_dim=lstm_hidden,
                                     n_qubits=lstm_qubits)

        # 5. Classical classifier head
        final_dim = lstm_hidden if use_lstm else auto_latent
        self.classifier = nn.Linear(final_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, 1, 28, 28) or (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (batch, num_classes).
        """
        if x.dim() == 5:
            batch, seq_len, c, h, w = x.shape
            x = x.view(batch * seq_len, c, h, w)
        else:
            seq_len = 1

        # 1. Quantum quanvolution: process each 2×2 patch
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [
                        x[:, r, c, 0],
                        x[:, r, c + 1, 0],
                        x[:, r + 1, c, 0],
                        x[:, r + 1, c + 1, 0],
                    ],
                    dim=1,
                )  # (batch*seq_len, 4)
                qfeat = self.qfilter(patch)  # (batch*seq_len, 4)
                patches.append(qfeat)
        features = torch.cat(patches, dim=1)  # (batch*seq_len, 4*14*14)

        # 2. QCNN block (placeholder: use the same feature dimension)
        qcnn_feat = self.qcnn(features)  # (batch*seq_len, 4)

        # 3. Quantum autoencoder encoding
        latent = self.autoencoder(qcnn_feat)  # (batch*seq_len, auto_latent)

        # 4. Sequence modelling
        if seq_len > 1:
            latent = latent.view(batch, seq_len, -1)
            latent, _ = self.lstm(latent)
            latent = latent[:, -1, :]

        # 5. Classification
        logits = self.classifier(latent)
        return nn.functional.log_softmax(logits, dim=-1)

    def decode_autoencoder(self, latent: torch.Tensor) -> torch.Tensor:
        """Reconstruct features from latent representation (quantum decoder)."""
        # For brevity, this is a placeholder that returns zeros of the original shape.
        return torch.zeros_like(latent)
