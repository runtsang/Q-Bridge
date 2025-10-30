import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class QuanvolutionFilterQuantum(tq.QuantumModule):
    """
    Quantum kernel that processes a 4‑dimensional patch.
    """
    def __init__(self, n_wires: int = 4, n_ops: int = 8):
        super().__init__()
        self.n_wires = n_wires
        self.n_ops = n_ops
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.random_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor of shape (B, 4)
        :return: Tensor of shape (B, 4)
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)
        self.encoder(qdev, x)
        self.random_layer(qdev)
        return self.measure(qdev)


class QuantumLSTM(tq.QuantumModule):
    """
    LSTM cell where each gate is realised by a small quantum circuit.
    """
    class QGate(tq.QuantumModule):
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
            qdev = tq.QuantumDevice(self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for w, gate in enumerate(self.params):
                gate(qdev, wires=w)
            # Entangle neighbouring wires
            for w in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[w, w + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    def __init__(self, n_input: int, n_hidden: int, n_layers: int = 1):
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.gates = nn.ModuleList()
        for _ in range(n_layers):
            self.gates.append({
                'f': self.QGate(n_hidden),
                'i': self.QGate(n_hidden),
                'g': self.QGate(n_hidden),
                'o': self.QGate(n_hidden)
            })
        self.linear_f = nn.Linear(n_input + n_hidden, n_hidden)
        self.linear_i = nn.Linear(n_input + n_hidden, n_hidden)
        self.linear_g = nn.Linear(n_input + n_hidden, n_hidden)
        self.linear_o = nn.Linear(n_input + n_hidden, n_hidden)

    def forward(self, x: torch.Tensor, states: tuple = None) -> tuple:
        """
        :param x: Tensor of shape (T, B, feature_dim)
        :return: (outputs, (hx, cx))
        """
        hx, cx = self._init_states(x, states)
        outputs = []
        for t in range(x.size(0)):
            combined = torch.cat([x[t], hx], dim=1)
            f = torch.sigmoid(self.gates[0]['f'](self.linear_f(combined)))
            i = torch.sigmoid(self.gates[0]['i'](self.linear_i(combined)))
            g = torch.tanh(self.gates[0]['g'](self.linear_g(combined)))
            o = torch.sigmoid(self.gates[0]['o'](self.linear_o(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self, x: torch.Tensor, states: tuple = None) -> tuple:
        if states is not None:
            return states
        batch = x.size(1)
        device = x.device
        return (torch.zeros(batch, self.n_hidden, device=device),
                torch.zeros(batch, self.n_hidden, device=device))
