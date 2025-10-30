import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple

class ClassicalQLSTM(nn.Module):
    """Classical LSTM cell mimicking the quantum interface."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
        return torch.zeros(batch_size, self.hidden_dim, device=device), torch.zeros(batch_size, self.hidden_dim, device=device)

class HybridQCNNQLSTM(nn.Module):
    """Hybrid classical model combining a QCNN‑style CNN backbone with a classical LSTM."""
    def __init__(self, in_features: int, hidden_dim: int, n_qubits: int = 0):
        super().__init__()
        # CNN backbone mimicking QCNN
        self.cnn = nn.Sequential(
            nn.Linear(in_features, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(),
            nn.Linear(16, 12), nn.Tanh(),
            nn.Linear(12, 8), nn.Tanh(),
            nn.Linear(8, 4), nn.Tanh(),
            nn.Linear(4, 4), nn.Tanh(),
            nn.Linear(4, 1)
        )
        # LSTM
        if n_qubits > 0:
            self.lstm = ClassicalQLSTM(in_features, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(in_features, hidden_dim, batch_first=True)
        # Final classifier
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, features).

        Returns
        -------
        cnn_features : torch.Tensor
            CNN feature map output for each timestep.
        logits : torch.Tensor
            Final classification logits from the LSTM.
        """
        batch, seq_len, _ = x.shape
        # Flatten sequence for CNN
        flat = x.reshape(batch * seq_len, -1)
        cnn_features = self.cnn(flat).reshape(batch, seq_len, -1)
        # Feed CNN features to LSTM
        if isinstance(self.lstm, ClassicalQLSTM):
            lstm_out, _ = self.lstm(cnn_features.permute(1, 0, 2))
            lstm_last = lstm_out[-1]
        else:
            lstm_out, _ = self.lstm(cnn_features)
            lstm_last = lstm_out[:, -1, :]
        logits = self.head(lstm_last)
        return cnn_features, logits

__all__ = ["HybridQCNNQLSTM", "ClassicalQLSTM"]
