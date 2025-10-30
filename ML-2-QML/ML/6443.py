import torch
import torch.nn as nn
import torch.nn.functional as F

class QCNNModel(nn.Module):
    """Classical QCNN‑inspired feature extractor."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

class QLSTM(nn.Module):
    """Classical LSTM cell with linear gates."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
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
            # Quantum path
            from.qml_module import QCNNQuantum, QLSTMQuantum  # lazy import
            self.cnn = QCNNQuantum(n_qubits)
            self.lstm = QLSTMQuantum(1, hidden_dim, n_qubits)
        else:
            # Classical path
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

__all__ = ["QCNNModel", "QLSTM", "HybridQLSTM"]
