import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ------------------------------------------------------------------
# Classical convolutional front‑end (drop‑in for quanvolution)
# ------------------------------------------------------------------
class ConvFilter(nn.Module):
    """Simple 1‑D convolution that mimics the quantum filter."""
    def __init__(self, kernel_size: int = 3, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, seq_len, features).  Only the first
            feature channel is used for the filter.
        Returns
        -------
        torch.Tensor
            Filtered output of shape (batch, seq_len - kernel_size + 1, 1)
        """
        # Collapse feature dimension and apply 1‑D conv
        x = x[:, :, :1].transpose(1, 2)          # (batch, 1, seq_len)
        out = self.conv(x)                       # (batch, 1, L_out)
        out = torch.sigmoid(out - self.threshold)
        return out.transpose(1, 2)                # (batch, L_out, 1)

# ------------------------------------------------------------------
# Classical hybrid head (mimics quantum expectation head)
# ------------------------------------------------------------------
class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid that replaces the quantum expectation."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class Hybrid(nn.Module):
    """Dense head that replaces a quantum circuit."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.linear(inputs)
        return HybridFunction.apply(logits, self.shift)

# ------------------------------------------------------------------
# Classical LSTM with optional quantum‑style gates
# ------------------------------------------------------------------
class QLayer(nn.Module):
    """
    Classical gate that imitates a quantum layer.
    The forward pass applies a linear transformation followed by
    a sigmoid activation – the same functional form used in the
    original quantum implementation.
    """
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))

class QLSTMHybrid(nn.Module):
    """
    Drop‑in replacement for the original QLSTM.
    Combines a classical convolutional front‑end, an LSTM with
    optional classical “quantum” gates, and a hybrid head for
    binary classification.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        conv_kernel: int = 3,
        use_hybrid: bool = True,
    ) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size=conv_kernel)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        # LSTM core – can be swapped with a quantum‑style gate set
        if n_qubits > 0:
            self.forget = QLayer(n_qubits)
            self.input = QLayer(n_qubits)
            self.update = QLayer(n_qubits)
            self.output = QLayer(n_qubits)
            self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)
            # LSTM cell is implemented manually
        else:
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, 1)
        self.hybrid = Hybrid(1, shift=0.0) if use_hybrid else None

    def _init_states(self, batch_size: int, device: torch.device):
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, features).

        Returns
        -------
        torch.Tensor
            Binary logits of shape (batch, seq_len, 2) if hybrid head
            is used, otherwise raw logits.
        """
        # Convolutional filtering
        x = self.conv(x)  # (batch, L_out, 1)

        batch_size, seq_len, _ = x.size()
        device = x.device

        if hasattr(self, 'lstm'):
            # Classical LSTM path
            out, _ = self.lstm(x)
        else:
            # Manual LSTM cell with “quantum” gates
            hx, cx = self._init_states(batch_size, device)
            outputs = []
            for t in range(seq_len):
                xt = x[:, t, :]  # (batch, 1)
                combined = torch.cat([xt, hx], dim=1)
                f = torch.sigmoid(self.forget(self.linear_forget(combined)))
                i = torch.sigmoid(self.input(self.linear_input(combined)))
                g = torch.tanh(self.update(self.linear_update(combined)))
                o = torch.sigmoid(self.output(self.linear_output(combined)))
                cx = f * cx + i * g
                hx = o * torch.tanh(cx)
                outputs.append(hx.unsqueeze(1))
            out = torch.cat(outputs, dim=1)

        logits = self.fc(out)  # (batch, seq_len, 1)
        if self.hybrid:
            logits = self.hybrid(logits)
            return torch.cat((logits, 1 - logits), dim=-1)
        return logits

__all__ = ["QLSTMHybrid"]
