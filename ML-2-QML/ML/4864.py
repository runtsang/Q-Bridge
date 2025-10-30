import torch
from torch import nn
import torch.nn.functional as F

class SharedFeatureExtractor(nn.Module):
    """Reusable linear feature extractor."""
    def __init__(self, in_dim: int = 8, hidden: int = 16):
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(x))

class QCNNModel(nn.Module):
    """QCNN-inspired stack of fully connected layers."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

class SamplerQNN(nn.Module):
    """Softmax sampler network with a small linear backbone."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)

class QLSTM(nn.Module):
    """Classical LSTM cell equivalent to the quantum version."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
    def forward(self, inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
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
    def _init_states(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class LSTMTagger(nn.Module):
    """Sequence tagging model that switches between classical and quantum LSTM."""
    def __init__(self, input_dim: int, hidden_dim: int, tagset_size: int, n_qubits: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        if n_qubits > 0:
            self.lstm = QLSTM(input_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: batch x seq_len x input_dim
        lstm_out, _ = self.lstm(inputs)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=-1)

class QCNNQLSTM_QSampler(nn.Module):
    """Unified hybrid model combining QCNN, a quantum LSTM, and a sampler."""
    def __init__(self, tagset_size: int = 10, n_qubits: int = 4):
        super().__init__()
        self.feature_extractor = SharedFeatureExtractor()
        self.qcnn = QCNNModel()
        self.lstm_tagger = LSTMTagger(input_dim=8, hidden_dim=16, tagset_size=tagset_size, n_qubits=n_qubits)
        self.sampler = SamplerQNN()
    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # inputs: batch x 8
        features = self.feature_extractor(inputs)
        qcnn_logits = self.qcnn(features)
        seq_input = features.unsqueeze(1)  # batch x 1 x 8
        tag_logits = self.lstm_tagger(seq_input)
        sample_dist = self.sampler(features)
        return qcnn_logits, tag_logits, sample_dist

__all__ = ["QCNNQLSTM_QSampler"]
