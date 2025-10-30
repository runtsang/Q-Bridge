import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

class ClassicalFCL(nn.Module):
    """Classical fully‑connected layer mimicking the quantum example."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        values = thetas.view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation

class ClassicalQLSTM(nn.Module):
    """Pure PyTorch LSTM cell with linear gates."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor,
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
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
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self, inputs: torch.Tensor,
                     states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class ClassicalSelfAttention(nn.Module):
    """Classical self‑attention block."""
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, rotation_params: torch.Tensor,
                entangle_params: torch.Tensor,
                inputs: torch.Tensor) -> torch.Tensor:
        query = inputs @ rotation_params.reshape(self.embed_dim, -1)
        key = inputs @ entangle_params.reshape(self.embed_dim, -1)
        value = inputs
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return scores @ value

class ClassicalQCNNModel(nn.Module):
    """Stack of fully‑connected layers emulating a quantum convolution."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

class HybridLayer(nn.Module):
    """Unified interface that chains FCL → LSTM → Self‑Attention → QCNN."""
    def __init__(self,
                 n_features: int = 1,
                 input_dim: int = 4,
                 hidden_dim: int = 8,
                 n_qubits: int = 4,
                 embed_dim: int = 4) -> None:
        super().__init__()
        self.fcl = ClassicalFCL(n_features)
        self.lstm = ClassicalQLSTM(input_dim, hidden_dim, n_qubits)
        self.attention = ClassicalSelfAttention(embed_dim)
        self.cnn = ClassicalQCNNModel()

    def forward(self,
                thetas: torch.Tensor,
                lstm_input: torch.Tensor,
                lstm_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                rotation_params: torch.Tensor = None,
                entangle_params: torch.Tensor = None,
                attention_input: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor,
                                                               torch.Tensor, torch.Tensor,
                                                               Tuple[torch.Tensor, torch.Tensor]]:
        fcl_out = self.fcl(thetas)
        lstm_out, lstm_states = self.lstm(lstm_input, lstm_states)
        attention_out = self.attention(rotation_params, entangle_params, attention_input)
        cnn_out = self.cnn(attention_out)
        return fcl_out, lstm_out, attention_out, cnn_out, lstm_states

__all__ = ["HybridLayer"]
