import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class QCNNQuantum(nn.Module):
    """Quantum version of the QCNN feature extractor."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            # Simple encoder that maps classical bits to rotations
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
            return self.measure(qdev)

    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        # Construct layers of the QCNN
        self.conv1 = self.QLayer(n_qubits)
        self.pool1 = self.QLayer(n_qubits)
        self.conv2 = self.QLayer(n_qubits)
        self.pool2 = self.QLayer(n_qubits)
        self.conv3 = self.QLayer(n_qubits)
        self.head = nn.Linear(n_qubits, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_qubits) – each qubit is fed a classical bit
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

class QLSTMQuantum(nn.Module):
    """Quantum LSTM cell where each gate is a small quantum circuit."""
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
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        # Gates
        self.forget = self.QLayer(n_qubits)
        self.input_gate = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output_gate = self.QLayer(n_qubits)
        # Linear projections to qubit space
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output_gate(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    @staticmethod
    def _init_states(
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, inputs.size(-1), device=device)
        cx = torch.zeros(batch_size, inputs.size(-1), device=device)
        return hx, cx

class HybridQLSTM(nn.Module):
    """Hybrid sequence‑tagger that can operate in classical or quantum mode."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.n_qubits = n_qubits
        if n_qubits > 0:
            self.cnn = QCNNQuantum(n_qubits)
            self.lstm = QLSTMQuantum(1, hidden_dim, n_qubits)
        else:
            self.cnn = QCNNModel()
            self.lstm = nn.LSTM(1, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, embed_dim)
        seq_len, batch_size, _ = embeds.size()
        cnn_out = []
        for t in range(seq_len):
            x = embeds[t]  # (batch, embed_dim)
            feat = self.cnn(x)  # (batch, 1)
            cnn_out.append(feat)
        cnn_out = torch.stack(cnn_out, dim=0)  # (seq_len, batch, 1)
        lstm_out, _ = self.lstm(cnn_out)
        tag_logits = self.hidden2tag(lstm_out.squeeze(-1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QCNNQuantum", "QLSTMQuantum", "HybridQLSTM"]
