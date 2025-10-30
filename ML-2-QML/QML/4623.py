import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumConvFilter(tq.QuantumModule):
    """Quantum emulation of a convolution filter that processes a 2×2 patch."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.5, n_qubits: int = 4) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.threshold = threshold
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # data shape: (batch, n_qubits)
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=data.shape[0], device=data.device)
        self.encoder(qdev, data)
        out = self.measure(qdev)
        probs = (out + 1) / 2  # Convert Pauli‑Z eigenvalues -1/1 to probabilities
        return probs.mean(dim=-1)

class QuantumLSTM(nn.Module):
    """LSTM cell where each gate is realised by a small variational quantum circuit."""
    class QGate(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for w, gate in enumerate(self.params):
                gate(qdev, wires=w)
            for w in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[w, w + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = self.QGate(n_qubits)
        self.input = self.QGate(n_qubits)
        self.update = self.QGate(n_qubits)
        self.output = self.QGate(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: tuple = None):
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

    def _init_states(self, inputs: torch.Tensor, states: tuple = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class UnifiedModel(tq.QuantumModule):
    """
    Hybrid quantum‑classical model that mirrors the classical ``UnifiedModel``.
    The convolution‑FC backbone is identical; the hidden LSTM stack is replaced by a
    quantum LSTM cell.  All tensors are kept on the same device for seamless back‑prop.
    """
    def __init__(self, n_qubits: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)
        self.lstm = QuantumLSTM(input_dim=4, hidden_dim=4, n_qubits=n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        feat = self.features(x)
        flat = feat.view(batch, -1)
        out = self.fc(flat)
        out = self.norm(out)

        seq = out.unsqueeze(1)  # (batch, seq_len=1, 4)
        lstm_out, _ = self.lstm(seq)
        return lstm_out.squeeze(1)

__all__ = ["QuantumConvFilter", "QuantumLSTM", "UnifiedModel"]
