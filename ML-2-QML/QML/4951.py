import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from typing import Tuple

class UnifiedQuantumHybridModel(nn.Module):
    """
    Hybrid model that extends the classical variant with quantum sub‑modules:
      * Same CNN+FC backbone as in the classical version.
      * A variational quantum LSTM cell where each gate is a small quantum circuit.
      * A quantum kernel to compute pairwise similarity between hidden states.
    The interface matches the classical implementation.
    """
    def __init__(self,
                 n_channels: int = 1,
                 n_classes: int = 4,
                 hidden_dim: int = 64,
                 seq_len: int = 1,
                 n_qubits: int = 4,
                 kernel_gamma: float = 1.0):
        super().__init__()
        self.n_qubits = n_qubits
        self.seq_len = seq_len

        # CNN backbone (shared with classical)
        self.backbone = nn.Sequential(
            nn.Conv2d(n_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(16 * 7 * 7, hidden_dim),
            nn.ReLU()
        )

        # Quantum LSTM cell
        self.lstm = self._build_qlstm_cell(hidden_dim, hidden_dim, n_qubits)

        # Classification head
        self.classifier = nn.Linear(hidden_dim, n_classes)
        self.norm = nn.BatchNorm1d(n_classes)

        # Quantum kernel
        self.kernel = self.Kernel()
        self.kernel_gamma = kernel_gamma

    # ----------------------------------------------
    # Quantum LSTM cell implementation
    # ----------------------------------------------
    class QLayer(tq.QuantumModule):
        """
        Variational circuit used for each LSTM gate.
        """
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.param_list = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        @tq.static_support
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for gate in self.param_list:
                gate(qdev)
            for i in range(self.n_wires - 1):
                tq.cnot(qdev, wires=[i, i + 1])
            return self.measure(qdev)

    def _build_qlstm_cell(self, input_dim: int, hidden_dim: int, n_qubits: int) -> nn.Module:
        """Return a quantum LSTM cell with four QLayer gates."""
        class QuantumLSTMCell(nn.Module):
            def __init__(self, input_dim, hidden_dim, n_qubits):
                super().__init__()
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                self.n_qubits = n_qubits

                self.forget_proj = nn.Linear(input_dim + hidden_dim, n_qubits)
                self.input_proj = nn.Linear(input_dim + hidden_dim, n_qubits)
                self.update_proj = nn.Linear(input_dim + hidden_dim, n_qubits)
                self.output_proj = nn.Linear(input_dim + hidden_dim, n_qubits)

                self.forget_gate = UnifiedQuantumHybridModel.QLayer(n_qubits)
                self.input_gate = UnifiedQuantumHybridModel.QLayer(n_qubits)
                self.update_gate = UnifiedQuantumHybridModel.QLayer(n_qubits)
                self.output_gate = UnifiedQuantumHybridModel.QLayer(n_qubits)

            def forward(self, x: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                combined = torch.cat([x, hx], dim=1)
                f = torch.sigmoid(self.forget_gate(self.forget_proj(combined)))
                i = torch.sigmoid(self.input_gate(self.input_proj(combined)))
                g = torch.tanh(self.update_gate(self.update_proj(combined)))
                o = torch.sigmoid(self.output_gate(self.output_proj(combined)))
                new_c = f * cx + i * g
                new_h = o * torch.tanh(new_c)
                return new_h, new_c

        return QuantumLSTMCell(input_dim, hidden_dim, n_qubits)

    # ----------------------------------------------
    # Quantum kernel implementation
    # ----------------------------------------------
    class Kernel(tq.QuantumModule):
        """
        Fixed quantum kernel ansatz.
        """
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
            self.ansatz = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "ry", "wires": [0]},
                    {"input_idx": [1], "func": "ry", "wires": [1]},
                    {"input_idx": [2], "func": "ry", "wires": [2]},
                    {"input_idx": [3], "func": "ry", "wires": [3]},
                ]
            )

        @tq.static_support
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            self.q_device.reset_states(x.shape[0])
            self.ansatz(self.q_device, x)
            for info in reversed(self.ansatz.op_list):
                func = tq.func_name_dict[info["func"]]
                params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
                func(self.q_device, wires=info["wires"], params=params)
            return torch.abs(self.q_device.states.view(-1)[0])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Tensor of shape (seq_len, batch, channels, H, W)
        Returns:
            logits: (batch, n_classes)
            kernel: (batch, batch) quantum kernel matrix
            states: (h_n, c_n) from quantum LSTM
        """
        seq_len, batch, _, _, _ = x.shape

        # Feature extraction per time step
        features = []
        for t in range(seq_len):
            feat = self.backbone(x[t])          # (batch, 16, 7, 7)
            feat = feat.view(batch, -1)         # (batch, 16*7*7)
            feat = self.feature_proj(feat)      # (batch, hidden_dim)
            features.append(feat.unsqueeze(0))  # (1, batch, hidden_dim)
        features = torch.cat(features, dim=0)    # (seq_len, batch, hidden_dim)

        # Quantum LSTM
        hx = torch.zeros(batch, self.lstm.hidden_dim, device=x.device)
        cx = torch.zeros(batch, self.lstm.hidden_dim, device=x.device)
        for t in range(seq_len):
            hx, cx = self.lstm(features[t], hx, cx)
        hidden = hx  # (batch, hidden_dim)

        # Classification
        logits = self.classifier(hidden)
        logits = self.norm(logits)
        logits = F.log_softmax(logits, dim=1)

        # Quantum kernel matrix
        kernel = torch.zeros(batch, batch, device=x.device)
        for i in range(batch):
            for j in range(batch):
                kernel[i, j] = self.kernel(hidden[i:i+1], hidden[j:j+1])

        return logits, kernel, (hx, cx)
