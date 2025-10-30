import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HybridFunction(torch.autograd.Function):
    """Sigmoid head that mimics a quantum expectation layer."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        out = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (out,) = ctx.saved_tensors
        return grad_output * out * (1 - out), None


class HybridHead(nn.Module):
    """Linear layer followed by the differentiable sigmoid."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(self.linear(x), self.shift)


class QCNNModel(nn.Module):
    """Fully‑connected analogue of a QCNN: feature map → conv → pool → conv → pool → conv → head."""
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


class QLSTMGate(nn.Module):
    """Quantum‑inspired LSTM gate implemented with linear layers."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output = nn.Linear(input_dim + hidden_dim, n_qubits)

        self.d_f = nn.Linear(n_qubits, hidden_dim)
        self.d_i = nn.Linear(n_qubits, hidden_dim)
        self.d_g = nn.Linear(n_qubits, hidden_dim)
        self.d_o = nn.Linear(n_qubits, hidden_dim)

    def forward(self, combined: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f = torch.sigmoid(self.d_f(self.forget(combined)))
        i = torch.sigmoid(self.d_i(self.input(combined)))
        g = torch.tanh(self.d_g(self.update(combined)))
        o = torch.sigmoid(self.d_o(self.output(combined)))
        return f, i, g, o


class EstimatorQNN(nn.Module):
    """Unified estimator that can operate in classical or quantum‑inspired mode."""
    def __init__(self,
                 input_dim: int = 8,
                 hidden_dim: int = 8,
                 n_qubits: int = 0,
                 sequence: bool = False) -> None:
        super().__init__()
        self.sequence = sequence
        self.n_qubits = n_qubits

        if sequence:
            self.lstm_gate = QLSTMGate(input_dim, hidden_dim, n_qubits)
        else:
            self.cnn = QCNNModel()

        self.head = HybridHead(1, shift=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, features) for CNN mode,
            or (batch, seq_len, features) for sequence mode.
        """
        if self.sequence:
            # Initialise hidden and cell states
            batch_size = x.size(0)
            device = x.device
            hx = torch.zeros(batch_size, self.lstm_gate.d_f.out_features, device=device)
            cx = torch.zeros_like(hx)

            outputs = []
            for t in range(x.size(1)):
                combined = torch.cat([x[:, t, :], hx], dim=1)
                f, i, g, o = self.lstm_gate(combined)
                cx = f * cx + i * g
                hx = o * torch.tanh(cx)
                outputs.append(hx.unsqueeze(1))
            out = torch.cat(outputs, dim=1)
            out = out[:, -1, :]  # last timestep
        else:
            out = self.cnn(x)

        out = self.head(out)
        return torch.cat((out, 1 - out), dim=-1)


__all__ = ["EstimatorQNN"]
