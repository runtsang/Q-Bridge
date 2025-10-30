import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuanvolutionFilter(tq.QuantumModule):
    """Quantum convolutional front‑end inspired by the original Quanvolution."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class QLSTM(nn.Module):
    """Quantum‑enhanced LSTM cell where each gate is a small variational circuit."""
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
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(self.n_wires, bsz=x.shape[0], device=x.device)
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

    def forward(self, inputs: torch.Tensor, states: tuple | None = None):
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
        states: tuple | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class HybridQLSTM(nn.Module):
    """Hybrid model that chains a quantum convolution front‑end with a quantum LSTM core."""
    def __init__(self,
                 hidden_dim: int,
                 tagset_size: int,
                 n_qubits: int,
                 conv_channels: int = 4,
                 conv_kernel: int = 2,
                 conv_stride: int = 2,
                 image_size: int = 28,
                 batch_first: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        self.n_qubits = n_qubits
        self.batch_first = batch_first

        # Quantum front‑end
        self.qfilter = QuanvolutionFilter()
        self.flatten = nn.Flatten()

        conv_out = image_size // conv_stride
        feature_dim = conv_channels * conv_out * conv_out

        # Quantum LSTM core
        self.lstm = QLSTM(feature_dim, hidden_dim, n_qubits)

        # Output head
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self,
                images: torch.Tensor,
                states: tuple | None = None) -> torch.Tensor:
        """
        `images` expected shape: ``(batch, seq_len, 1, H, W)`` or ``(seq_len, batch, 1, H, W)``
        depending on ``batch_first``.  The method runs each frame through the
        quantum convolution, flattens the result, feeds it to the quantum LSTM,
        and finally maps the hidden states to tag logits.
        """
        if self.batch_first:
            batch, seq_len, c, h, w = images.shape
            x = images.view(batch * seq_len, c, h, w)
        else:
            seq_len, batch, c, h, w = images.shape
            x = images.view(seq_len * batch, c, h, w)

        # Quantum convolution
        x = self.qfilter(x)
        x = self.flatten(x)  # shape (batch*seq_len, feature_dim)

        # Restore sequence structure
        if self.batch_first:
            x = x.view(batch, seq_len, -1)
        else:
            x = x.view(seq_len, batch, -1)

        # Quantum LSTM core
        lstm_out, _ = self.lstm(x, states)

        # Tagging head
        logits = self.hidden2tag(lstm_out)
        return F.log_softmax(logits, dim=-1)

# Backwards compatibility aliases
LSTMTagger = HybridQLSTM

__all__ = ["QLSTM", "LSTMTagger", "HybridQLSTM"]
