import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict, cnot as tq_cnot

class KernalAnsatz(tq.QuantumModule):
    """Quantum feature‑map that encodes two classical vectors."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class Kernel(tq.QuantumModule):
    """Fixed quantum kernel evaluated via a 4‑wire ansatz."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

class QLayer(tq.QuantumModule):
    """Parameter‑ized quantum layer that applies RX gates followed by a CNOT chain."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "rx", "wires": [0]},
                {"input_idx": [1], "func": "rx", "wires": [1]},
                {"input_idx": [2], "func": "rx", "wires": [2]},
                {"input_idx": [3], "func": "rx", "wires": [3]},
            ]
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
                tq_cnot(qdev, wires=[wire, 0])
            else:
                tq_cnot(qdev, wires=[wire, wire + 1])
        return self.measure(qdev)

class HybridQLSTM(nn.Module):
    """Hybrid LSTM cell that mixes quantum gates with a quantum kernel."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = hidden_dim  # Ensure gate and linear outputs match hidden_dim

        # Quantum gates for each LSTM gate
        self.forget_gate = QLayer(self.n_qubits)
        self.input_gate = QLayer(self.n_qubits)
        self.update_gate = QLayer(self.n_qubits)
        self.output_gate = QLayer(self.n_qubits)

        # Classical linear projections that feed the quantum gates
        self.forget_linear = nn.Linear(input_dim + hidden_dim, self.n_qubits)
        self.input_linear = nn.Linear(input_dim + hidden_dim, self.n_qubits)
        self.update_linear = nn.Linear(input_dim + hidden_dim, self.n_qubits)
        self.output_linear = nn.Linear(input_dim + hidden_dim, self.n_qubits)

        self.kernel = Kernel()

    def _gate(self, gate: QLayer, linear: nn.Linear,
              x: torch.Tensor, hx: torch.Tensor) -> torch.Tensor:
        lin_out = linear(torch.cat([x, hx], dim=1))
        k_val = self.kernel(x, hx).unsqueeze(1).repeat(1, self.hidden_dim)
        return torch.sigmoid(gate(lin_out) + k_val)

    def forward(self, inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            f = self._gate(self.forget_gate, self.forget_linear, x, hx)
            i = self._gate(self.input_gate, self.input_linear, x, hx)
            g = torch.tanh(self.update_linear(torch.cat([x, hx], dim=1)))
            o = self._gate(self.output_gate, self.output_linear, x, hx)
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self, inputs: torch.Tensor,
                     states: tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch_size, self.hidden_dim, device=device), \
               torch.zeros(batch_size, self.hidden_dim, device=device)

class HybridLSTMTagger(nn.Module):
    """Sequence tagging model built around the hybrid quantum‑classical LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int,
                 vocab_size: int, tagset_size: int,
                 n_qubits: int = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return torch.nn.functional.log_softmax(tag_logits, dim=1)

__all__ = ["HybridQLSTM", "HybridLSTMTagger"]
