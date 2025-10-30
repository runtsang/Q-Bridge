import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.transpiler import transpile
from qiskit.compiler import assemble

from.ClassicalQuantumBinaryClassification__gen325 import HybridBinaryQLSTMNet as ClassicalHybridBinaryQLSTMNet

class QuantumExpectation(nn.Module):
    """Map a scalar to a quantum expectation via a 1‑qubit circuit."""
    def __init__(self, shots: int = 256):
        super().__init__()
        self.shots = shots
        self.backend = Aer.get_backend('aer_simulator')
        self.theta = Parameter('θ')
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(self.theta, 0)
        qc.measure_all()
        self.circuit = qc
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        thetas = x.detach().cpu().numpy()
        exps = []
        for theta in thetas:
            bound = self.circuit.bind_parameters({self.theta: theta})
            transpiled = transpile(bound, self.backend)
            qobj = assemble(transpiled, shots=self.shots)
            result = execute(qobj, self.backend).result()
            counts = result.get_counts()
            exp = sum(int(bit, 2) * (cnt / self.shots) for bit, cnt in counts.items())
            exps.append(exp)
        return torch.tensor(exps, device=x.device, dtype=x.dtype)

class QuantumGateLayer(nn.Module):
    """Linear projection followed by a quantum expectation."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.qexp = QuantumExpectation()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = self.linear(x)
        return self.qexp(proj)

class QuantumLSTMCell(nn.Module):
    """LSTM cell where each gate is a QuantumGateLayer."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.forget = QuantumGateLayer(input_dim + hidden_dim, hidden_dim)
        self.input = QuantumGateLayer(input_dim + hidden_dim, hidden_dim)
        self.update = QuantumGateLayer(input_dim + hidden_dim, hidden_dim)
        self.output = QuantumGateLayer(input_dim + hidden_dim, hidden_dim)
    def forward(self, x: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor):
        combined = torch.cat([x, hx], dim=-1)
        f = torch.sigmoid(self.forget(combined))
        i = torch.sigmoid(self.input(combined))
        g = torch.tanh(self.update(combined))
        o = torch.sigmoid(self.output(combined))
        cx = f * cx + i * g
        hx = o * torch.tanh(cx)
        return hx, cx

class QuantumLSTM(nn.Module):
    """Stacked QuantumLSTMCell for sequence tagging."""
    def __init__(self, input_dim: int, hidden_dim: int, layers: int = 1, tagset: int = 10):
        super().__init__()
        self.layers = nn.ModuleList([QuantumLSTMCell(input_dim if i==0 else hidden_dim, hidden_dim)
                                     for i in range(layers)])
        self.fc = nn.Linear(hidden_dim, tagset)
    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = seq.shape
        h = [torch.zeros(batch, self.layers[0].hidden_dim, device=seq.device)
             for _ in range(len(self.layers))]
        c = [torch.zeros(batch, self.layers[0].hidden_dim, device=seq.device)
             for _ in range(len(self.layers))]
        outputs = []
        for t in range(seq_len):
            x = seq[:, t, :]
            for l, cell in enumerate(self.layers):
                h[l], c[l] = cell(x, h[l], c[l])
                x = h[l]
            outputs.append(x.unsqueeze(1))
        out = torch.cat(outputs, dim=1)
        return F.log_softmax(self.fc(out), dim=-1)

class HybridBinaryQLSTMNet(ClassicalHybridBinaryQLSTMNet):
    """Quantum‑augmented hybrid model that replaces the head and LSTM with quantum modules."""
    def __init__(self, use_lstm: bool = True, lstm_input: int = 32*4*4, lstm_hidden: int = 64):
        super().__init__(use_lstm=False)
        self.head = QuantumGateLayer(self.backbone.flatten_size, 1)
        if use_lstm:
            self.lstm = QuantumLSTM(lstm_input, lstm_hidden)
        else:
            self.lstm = None
    def forward(self, image: torch.Tensor, seq: torch.Tensor | None = None) -> torch.Tensor | tuple:
        feat = self.backbone(image)
        logits = self.head(feat)
        probs = torch.cat((logits, 1 - logits), dim=-1)
        if self.lstm and seq is not None:
            tag_out = self.lstm(seq)
            return probs, tag_out
        return probs

__all__ = ["HybridBinaryQLSTMNet", "QuantumExpectation", "QuantumGateLayer", "QuantumLSTM", "QuantumLSTMCell"]
