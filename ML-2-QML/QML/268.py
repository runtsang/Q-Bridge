import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QLSTM(nn.Module):
    """Quantum‑enhanced LSTM where each gate is a variational quantum circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int, circuit_depth: int = 2, entanglement: str = "chain"):
            super().__init__()
            self.n_wires = n_wires
            self.circuit_depth = circuit_depth
            self.entanglement = entanglement
            # Trainable rotation parameters: depth × wires × 3 (Rx,Ry,Rz)
            self.params = nn.Parameter(torch.randn(circuit_depth, n_wires, 3))
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            for depth in range(self.circuit_depth):
                for wire in range(self.n_wires):
                    rx = self.params[depth, wire, 0]
                    ry = self.params[depth, wire, 1]
                    rz = self.params[depth, wire, 2]
                    tq.RX(rx)(qdev, wires=wire)
                    tq.RY(ry)(qdev, wires=wire)
                    tq.RZ(rz)(qdev, wires=wire)
                # Entangling layer
                if self.entanglement == "chain":
                    for wire in range(self.n_wires - 1):
                        tq.CNOT(qdev, wires=[wire, wire + 1])
                elif self.entanglement == "full":
                    for wire1 in range(self.n_wires):
                        for wire2 in range(wire1 + 1, self.n_wires):
                            tq.CNOT(qdev, wires=[wire1, wire2])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int,
                 circuit_depth: int = 2, entanglement: str = "chain"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.circuit_depth = circuit_depth
        self.entanglement = entanglement

        # Quantum layers for gates
        self.forget_gate = self.QLayer(n_qubits, circuit_depth, entanglement)
        self.input_gate = self.QLayer(n_qubits, circuit_depth, entanglement)
        self.update_gate = self.QLayer(n_qubits, circuit_depth, entanglement)
        self.output_gate = self.QLayer(n_qubits, circuit_depth, entanglement)

        # Linear projections to quantum space
        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Classical post‑processing head
        self.post_linear = nn.Linear(n_qubits, hidden_dim)

    def forward(self, inputs: torch.Tensor, states: tuple | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.forget_lin(combined)))
            i = torch.sigmoid(self.input_gate(self.input_lin(combined)))
            g = torch.tanh(self.update_gate(self.update_lin(combined)))
            o = torch.sigmoid(self.output_gate(self.output_lin(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        hidden_states = torch.cat(outputs, dim=0)  # [seq_len, batch, hidden_dim]
        # Classical post‑processing on quantum outputs
        processed = self.post_linear(hidden_states)
        return processed, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: tuple | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 0,
                 circuit_depth: int = 2, entanglement: str = "chain"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits,
                              circuit_depth=circuit_depth, entanglement=entanglement)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        embeds = self.word_embeddings(sentence)
        if isinstance(self.lstm, nn.LSTM):
            lstm_out, _ = self.lstm(embeds.unsqueeze(1))
            lstm_out = lstm_out.squeeze(1)
        else:
            lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
