import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import StatevectorSampler

# ------------------------------------------------------------------
# Quantum self‑attention circuit
# ------------------------------------------------------------------
class QuantumSelfAttention(tq.QuantumModule):
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder([
            {"input_idx": [0], "func": "rx", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "rz", "wires": [2]},
            {"input_idx": [3], "func": "rx", "wires": [3]},
        ])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_qubits,
                                bsz=x.shape[0],
                                device=x.device)
        self.encoder(qdev, x)
        return self.measure(qdev)

# ------------------------------------------------------------------
# Quantum quanvolution filter
# ------------------------------------------------------------------
class QuantumQuanvolutionFilter(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder([
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ])
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
                data = torch.stack([
                    x[:, r, c],
                    x[:, r, c + 1],
                    x[:, r + 1, c],
                    x[:, r + 1, c + 1],
                ], dim=1)
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

# ------------------------------------------------------------------
# Quantum sampler network (Qiskit)
# ------------------------------------------------------------------
class QuantumSamplerQNN:
    def __init__(self):
        inputs2 = ParameterVector("input", 2)
        weights2 = ParameterVector("weight", 4)
        qc2 = QuantumCircuit(2)
        qc2.ry(inputs2[0], 0)
        qc2.ry(inputs2[1], 1)
        qc2.cx(0, 1)
        qc2.ry(weights2[0], 0)
        qc2.ry(weights2[1], 1)
        qc2.cx(0, 1)
        qc2.ry(weights2[2], 0)
        qc2.ry(weights2[3], 1)
        sampler = StatevectorSampler()
        self.sampler_qnn = QiskitSamplerQNN(circuit=qc2,
                                            input_params=inputs2,
                                            weight_params=weights2,
                                            sampler=sampler)

    def sample(self, inputs: torch.Tensor) -> torch.Tensor:
        inp_np = inputs.detach().cpu().numpy()
        samples = self.sampler_qnn.sample(inp_np)
        return torch.tensor(samples, device=inputs.device, dtype=torch.float32)

# ------------------------------------------------------------------
# Quantum LSTM cell with parameterised gates
# ------------------------------------------------------------------
class QLayer(tq.QuantumModule):
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder([
            {"input_idx": [0], "func": "rx", "wires": [0]},
            {"input_idx": [1], "func": "rx", "wires": [1]},
            {"input_idx": [2], "func": "rx", "wires": [2]},
            {"input_idx": [3], "func": "rx", "wires": [3]},
        ])
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires,
                                bsz=x.shape[0],
                                device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        for wire in range(self.n_wires):
            if wire == self.n_wires - 1:
                tqf.cnot(qdev, wires=[wire, 0])
            else:
                tqf.cnot(qdev, wires=[wire, wire + 1])
        return self.measure(qdev)

class QLSTMQuantum(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = QLayer(n_qubits)
        self.input = QLayer(n_qubits)
        self.update = QLayer(n_qubits)
        self.output = QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self,
                inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None):
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
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch_size, self.hidden_dim, device=device), \
               torch.zeros(batch_size, self.hidden_dim, device=device)

# ------------------------------------------------------------------
# Hybrid LSTM‑tagger (quantum core)
# ------------------------------------------------------------------
class HybridQLSTM(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 use_quanvolution: bool = False,
                 use_self_attention: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Optional quantum‑inspired feature extractor
        self.use_quanvolution = use_quanvolution
        self.feature_extractor = QuantumQuanvolutionFilter() if use_quanvolution else nn.Identity()

        # LSTM core – quantum by default
        self.lstm = QLSTMQuantum(embedding_dim, hidden_dim, n_qubits=n_qubits) if n_qubits > 0 else nn.LSTM(embedding_dim, hidden_dim)

        # Optional quantum self‑attention
        self.use_self_attention = use_self_attention
        self.attention = QuantumSelfAttention(n_qubits) if use_self_attention else None

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.sampler = QuantumSamplerQNN()

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        if self.use_quanvolution:
            embeds = self.feature_extractor(embeds)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        if self.use_self_attention:
            lstm_out = self.attention(lstm_out)
        logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(logits, dim=1)

__all__ = ["HybridQLSTM", "QLSTMQuantum", "QuantumSelfAttention",
           "QuantumQuanvolutionFilter", "QuantumSamplerQNN"]
