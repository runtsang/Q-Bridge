import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumSelfAttention(nn.Module):
    """Variational self‑attention circuit implemented with Pennylane."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.device = qml.device("default.qubit", wires=embed_dim)
        self.rotation_params = nn.Parameter(torch.randn(embed_dim))
        self.entangle_params = nn.Parameter(torch.randn(embed_dim - 1))

        @qml.qnode(self.device, interface="torch")
        def circuit(token):
            for i in range(self.embed_dim):
                qml.RX(token[i] + self.rotation_params[i], wires=i)
            for i in range(self.embed_dim - 1):
                qml.CRY(self.entangle_params[i], wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.embed_dim)]

        self._qnode = circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = inputs.size()
        out = torch.empty_like(inputs)
        for b in range(batch):
            for t in range(seq_len):
                out[b, t] = self._qnode(inputs[b, t])
        return out.to(inputs.device)

class QuantumQLSTM(nn.Module):
    """LSTM cell where each gate is a small variational circuit via TorchQuantum."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                if wire == self.n_wires - 1:
                    tqf.cnot(qdev, wires=[wire, 0])
                else:
                    tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states=None):
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
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self, inputs, states):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class UnifiedSelfAttentionQLSTM(nn.Module):
    """Drop‑in quantum model combining a quantum self‑attention block and a quantum‑enhanced LSTM."""
    def __init__(self, embed_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.attention = QuantumSelfAttention(embed_dim)
        self.lstm = QuantumQLSTM(embed_dim, hidden_dim, n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, embed_dim)
        embeds_perm = embeds.permute(1, 0, 2)     # (batch, seq_len, embed_dim)
        attn_out = self.attention(embeds_perm)    # (batch, seq_len, embed_dim)
        attn_out = attn_out.permute(1, 0, 2)     # (seq_len, batch, embed_dim)
        lstm_out, _ = self.lstm(attn_out)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)

__all__ = ["UnifiedSelfAttentionQLSTM"]
