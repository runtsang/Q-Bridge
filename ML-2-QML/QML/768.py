import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QLSTM(nn.Module):
    """Quantum-enhanced LSTM with a shared variational quantum gate.

    The quantum circuit is executed once per time‑step and its output
    is fed into all four gates.  The circuit consists of a trainable
    RX layer followed by a chain of CNOTs that entangles the wires.
    """
    class SharedQLayer(tq.QuantumModule):
        def __init__(self, n_wires: int, depth: int = 2):
            super().__init__()
            self.n_wires = n_wires
            self.depth = depth
            # Parameter‑shared RX gates
            self.rx_params = nn.Parameter(torch.randn(n_wires))
            # Entangling pattern: chain + wrap‑around
            self.cnot_pattern = [(i, i+1) for i in range(n_wires-1)] + [(n_wires-1, 0)]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (batch, n_wires)
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            # Encode classical data
            for i in range(self.n_wires):
                tqf.rx(qdev, x[:, i], wires=i)
            # Apply variational layers
            for _ in range(self.depth):
                for wire in range(self.n_wires):
                    tqf.rx(qdev, self.rx_params[wire], wires=wire)
                for src, tgt in self.cnot_pattern:
                    tqf.cnot(qdev, wires=[src, tgt])
            # Measure all qubits
            return tq.MeasureAll(tq.PauliZ)(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, depth: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth

        self.shared_qgate = self.SharedQLayer(n_qubits, depth=depth)

        # Linear projections to quantum space
        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: tuple | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.shared_qgate(self.forget_lin(combined)))
            i = torch.sigmoid(self.shared_qgate(self.input_lin(combined)))
            g = torch.tanh(self.shared_qgate(self.update_lin(combined)))
            o = torch.sigmoid(self.shared_qgate(self.output_lin(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: tuple | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch_size, self.hidden_dim, device=device), torch.zeros(batch_size, self.hidden_dim, device=device)

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int,
                 n_qubits: int = 0, depth: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits, depth=depth)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
