import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QLSTM(tq.QuantumModule):
    """Quantum‑enhanced LSTM where each gate is a small variational circuit."""
    class QGate(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            # Encode each input feature into a rotation on its wire
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            bsz = x.shape[0]
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
            self.encoder(qdev, x)
            for gate, wire in zip(self.params, range(self.n_wires)):
                gate(qdev, wires=wire)
            # Entangle wires with a linear CNOT chain
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Quantum gates for each LSTM component
        self.forget_gate = self.QGate(n_qubits)
        self.input_gate = self.QGate(n_qubits)
        self.update_gate = self.QGate(n_qubits)
        self.output_gate = self.QGate(n_qubits)

        # Linear maps from classical concatenated vector to qubit space
        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Linear layers to map qubit probabilities to gate activations
        self.forget_out = nn.Linear(n_qubits, hidden_dim)
        self.input_out = nn.Linear(n_qubits, hidden_dim)
        self.update_out = nn.Linear(n_qubits, hidden_dim)
        self.output_out = nn.Linear(n_qubits, hidden_dim)

    def forward(self, inputs: torch.Tensor, states: tuple | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f_q = self.forget_gate(self.forget_lin(combined))
            i_q = self.input_gate(self.input_lin(combined))
            g_q = self.update_gate(self.update_lin(combined))
            o_q = self.output_gate(self.output_lin(combined))
            f = torch.sigmoid(self.forget_out(f_q))
            i = torch.sigmoid(self.input_out(i_q))
            g = torch.tanh(self.update_out(g_q))
            o = torch.sigmoid(self.output_out(o_q))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: tuple | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class QFCLayer(tq.QuantumModule):
    """Quantum fully‑connected head inspired by Quantum‑NAT."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.n_wires = in_features
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(in_features)]
        )
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(self.n_wires)))
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(self.n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear = nn.Linear(self.n_wires, out_features)

    def forward_classical(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        self.encoder(qdev, x)
        self.random_layer(qdev)
        for gate, wire in zip(self.params, range(self.n_wires)):
            gate(qdev, wires=wire)
        out = self.measure(qdev)
        return self.linear(out)

class LSTMTagger(nn.Module):
    """Sequence tagging model that uses the quantum LSTM core and quantum head."""
    def __init__(self, embedding_dim: int, hidden_dim: int,
                 vocab_size: int, tagset_size: int, n_qubits: int):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.qfc_head = QFCLayer(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, embedding_dim)
        lstm_out, _ = self.lstm(embeds)
        seq_len, batch, _ = lstm_out.shape
        flat = lstm_out.permute(1, 0, 2).reshape(batch * seq_len, -1)
        logits = self.qfc_head.forward_classical(flat)
        logits = logits.view(seq_len, batch, -1)
        return F.log_softmax(logits, dim=2)

__all__ = ["QLSTM", "LSTMTagger"]
