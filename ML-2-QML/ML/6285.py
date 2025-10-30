import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class ConvFilter(nn.Module):
    """Classical convolution filter that emulates a quantum quanvolution kernel."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = nn.Parameter(torch.tensor(threshold))
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
    def forward(self, x):
        # x shape: (batch, 1, H, W)
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        # average over spatial dimensions
        return activations.mean(dim=(2,3))

class QLayer(nn.Module):
    """Classical approximation to a quantum circuit used in gates."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.linear = nn.Linear(n_wires, n_wires)
        self.entangle = nn.Linear(n_wires, n_wires)
    def forward(self, x):
        # x shape: (batch, n_wires)
        out = torch.tanh(self.linear(x))
        out = out * torch.sigmoid(self.entangle(out))
        return out

class QLSTM(nn.Module):
    """Hybrid LSTM where each gate is a small quantum circuit (QLayer)."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget_gate = QLayer(n_qubits)
        self.input_gate = QLayer(n_qubits)
        self.update_gate = QLayer(n_qubits)
        self.output_gate = QLayer(n_qubits)
        self.fc_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.fc_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.fc_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.fc_output = nn.Linear(input_dim + hidden_dim, n_qubits)
    def forward(self, inputs, states=None):
        # inputs shape: (seq_len, batch, input_dim)
        batch_size = inputs.size(1)
        hx = torch.zeros(batch_size, self.hidden_dim, device=inputs.device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=inputs.device)
        if states is not None:
            hx, cx = states
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.fc_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.fc_input(combined)))
            g = torch.tanh(self.update_gate(self.fc_update(combined)))
            o = torch.sigmoid(self.output_gate(self.fc_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

class HybridConvQLSTM(nn.Module):
    """Drop‑in replacement for image‑based tagging that fuses ConvFilter and QLSTM."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.5,
                 input_dim: int = 1, hidden_dim: int = 32,
                 n_qubits: int = 4, num_classes: int = 10):
        super().__init__()
        self.conv_filter = ConvFilter(kernel_size, threshold)
        self.lstm = QLSTM(input_dim, hidden_dim, n_qubits)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    def forward(self, images):
        # images shape: (batch, seq_len, 1, H, W)
        batch, seq_len, _, H, W = images.shape
        conv_out = []
        for t in range(seq_len):
            patch = images[:, t, :, :, :]  # (batch, 1, H, W)
            feat = self.conv_filter(patch)  # (batch,)
            conv_out.append(feat.unsqueeze(-1))
        conv_seq = torch.cat(conv_out, dim=1)  # (batch, seq_len)
        lstm_input = conv_seq.unsqueeze(-1).permute(1, 0, 2)
        lstm_out, _ = self.lstm(lstm_input)
        logits = self.classifier(lstm_out)
        return logits
