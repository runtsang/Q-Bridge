import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Optional

# --------------------------------------------------------------------------- #
# Quantum feature encoder
# --------------------------------------------------------------------------- #
class _QuantumFeatureEncoder(tq.QuantumModule):
    """Variational circuit that encodes a classical vector into 4 qubits."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.rx0 = tq.RX(has_params=True, trainable=True)
        self.ry0 = tq.RY(has_params=True, trainable=True)
        self.rz0 = tq.RZ(has_params=True, trainable=True)
        self.crx0 = tq.CRX(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor) -> torch.Tensor:
        self.encoder(qdev, x)
        self.rx0(qdev, wires=0)
        self.ry0(qdev, wires=1)
        self.rz0(qdev, wires=3)
        self.crx0(qdev, wires=[0, 2])
        tqf.hadamard(qdev, wires=3)
        tqf.sx(qdev, wires=2)
        tqf.cnot(qdev, wires=[3, 0])
        return self.measure(qdev)

# --------------------------------------------------------------------------- #
# Quantum gate used inside the quantum LSTM cell
# --------------------------------------------------------------------------- #
class _QuantumGate(tq.QuantumModule):
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

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(qdev)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        for wire in range(self.n_wires):
            if wire == self.n_wires - 1:
                tqf.cnot(qdev, wires=[wire, 0])
            else:
                tqf.cnot(qdev, wires=[wire, wire + 1])
        return self.measure(qdev)

# --------------------------------------------------------------------------- #
# Quantum LSTM cell
# --------------------------------------------------------------------------- #
class _QuantumLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget_gate = _QuantumGate(n_qubits)
        self.input_gate = _QuantumGate(n_qubits)
        self.update_gate = _QuantumGate(n_qubits)
        self.output_gate = _QuantumGate(n_qubits)

        self.lin_f = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_i = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_u = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_o = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, x: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor) -> tuple:
        combined = torch.cat([x, hx], dim=1)
        f = torch.sigmoid(self.forget_gate(self.lin_f(combined)))
        i = torch.sigmoid(self.input_gate(self.lin_i(combined)))
        g = torch.tanh(self.update_gate(self.lin_u(combined)))
        o = torch.sigmoid(self.output_gate(self.lin_o(combined)))
        cx = f * cx + i * g
        hx = o * torch.tanh(cx)
        return hx, cx

# --------------------------------------------------------------------------- #
# Hybrid model
# --------------------------------------------------------------------------- #
class QuantumHybridNat(nn.Module):
    """
    Hybrid model that supports both image classification and sequence tagging.
    Image mode uses a classical CNN feature extractor followed by a quantum encoder.
    Sequence mode uses an embedding layer and a quantum LSTM cell.
    """
    def __init__(
        self,
        mode: str = "image",
        *,
        in_channels: int = 1,
        vocab_size: Optional[int] = None,
        embedding_dim: Optional[int] = None,
        hidden_dim: int = 128,
        tagset_size: Optional[int] = None,
        n_qubits: int = 4,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.n_qubits = n_qubits
        if mode == "image":
            self.extractor = nn.Sequential(
                nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.fc = nn.Sequential(
                nn.Linear(16 * 7 * 7, 128),
                nn.ReLU(),
                nn.Linear(128, n_qubits),
            )
            self.encoder = _QuantumFeatureEncoder()
            self.readout = nn.Linear(n_qubits, 4)
            self.bn = nn.BatchNorm1d(4)
        elif mode == "sequence":
            assert vocab_size is not None and embedding_dim is not None and tagset_size is not None
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm_cell = _QuantumLSTMCell(embedding_dim, hidden_dim, n_qubits)
            self.classifier = nn.Linear(hidden_dim, tagset_size)
        else:
            raise ValueError(f"Unsupported mode {mode!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "image":
            feats = self.extractor(x)
            flat = feats.view(x.shape[0], -1)
            projected = self.fc(flat)
            qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=projected.shape[0], device=projected.device)
            out = self.encoder(qdev, projected)
            out = self.readout(out)
            return self.bn(out)
        elif self.mode == "sequence":
            embeds = self.embedding(x)
            batch, seq_len, _ = embeds.shape
            hx = torch.zeros(batch, self.lstm_cell.hidden_dim, device=embeds.device)
            cx = torch.zeros(batch, self.lstm_cell.hidden_dim, device=embeds.device)
            outputs = []
            for t in range(seq_len):
                hx, cx = self.lstm_cell(embeds[:, t, :], hx, cx)
                outputs.append(hx.unsqueeze(1))
            lstm_out = torch.cat(outputs, dim=1)
            logits = self.classifier(lstm_out)
            return F.log_softmax(logits, dim=-1)
        else:
            raise RuntimeError("Forward called with unknown mode")

__all__ = ["QuantumHybridNat"]
