import torch
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum import QuantumModule
from torchquantum.functional import func_name_dict
import numpy as np
from typing import Tuple, Sequence

# --------------------- Quantum LSTM --------------------- #
class QuantumLSTM(QuantumModule):
    """
    LSTM where each gate is a small quantum circuit.
    """
    class QGate(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]}
                 for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RX() for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor) -> torch.Tensor:
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget = self.QGate(n_qubits)
        self.input_gate = self.QGate(n_qubits)
        self.update = self.QGate(n_qubits)
        self.output = self.QGate(n_qubits)
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self, inputs: torch.Tensor,
                     states: Tuple[torch.Tensor, torch.Tensor] | None
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch_size, self.hidden_dim, device=device), \
               torch.zeros(batch_size, self.hidden_dim, device=device)

# --------------------- Hybrid Convolution --------------------- #
class QuantumHybridConv(tq.QuantumModule):
    """
    Quanvolution layer that can be turned into a classical Conv2d.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0,
                 use_quantum: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.use_quantum = use_quantum
        if use_quantum:
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]}
                 for i in range(kernel_size**2)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)
        else:
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        if self.use_quantum:
            flat = data.view(data.size(0), -1)
            qdev = tq.QuantumDevice(n_wires=flat.size(1),
                                    bsz=data.size(0),
                                    device=data.device)
            self.encoder(qdev, flat)
            return self.measure(qdev).mean(dim=1)
        else:
            tensor = data.view(-1, 1, self.kernel_size, self.kernel_size)
            logits = self.conv(tensor)
            return torch.sigmoid(logits - self.threshold).mean(dim=(2, 3))

# --------------------- Quantum Kernel --------------------- #
class QuantumKernel(tq.QuantumModule):
    """
    Quantum kernel using a fixed RX‑RY circuit.
    """
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.qdev = tq.QuantumDevice(n_wires=n_wires)
        self.ansatz = tq.QuantumModule(
            lambda q, x, y: [
                tq.RY(x[:, i], wires=i) for i in range(n_wires)
            ] + [
                tq.RY(-y[:, i], wires=i) for i in range(n_wires)
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.qdev, x, y)
        return torch.abs(self.qdev.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                  n_wires: int = 4) -> torch.Tensor:
    kernel = QuantumKernel(n_wires=n_wires)
    return torch.stack([torch.stack([kernel(x, y) for y in b]) for x in a])

# --------------------- Hybrid LSTM Tagger --------------------- #
class HybridLSTMTagger(tq.QuantumModule):
    """
    Sequence tagging model that combines the quantum LSTM, a hybrid conv front‑end,
    and a quantum kernel head.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int,
                 vocab_size: int, tagset_size: int,
                 use_quantum_lstm: bool = True,
                 use_quantum_conv: bool = True,
                 use_quantum_kernel: bool = True):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QuantumLSTM(embedding_dim, hidden_dim, n_qubits=hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.conv = QuantumHybridConv(kernel_size=2,
                                      threshold=0.0,
                                      use_quantum=use_quantum_conv)
        self.kernel = QuantumKernel() if use_quantum_kernel else None

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.unsqueeze(1))
        logits = self.hidden2tag(lstm_out.squeeze(1))
        if self.kernel is not None:
            proto = nn.Parameter(torch.randn_like(logits))
            sim = self.kernel(logits, proto)
            logits = logits * sim
        return F.log_softmax(logits, dim=1)

__all__ = ["QuantumLSTM", "QuantumHybridConv", "QuantumKernel",
           "kernel_matrix", "HybridLSTMTagger"]
