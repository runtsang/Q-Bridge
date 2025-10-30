import torch
import torch.nn as nn
import torchquantum as tq
import numpy as np

class QLSTMGen384(tq.QuantumModule):
    """Quantum LSTM cell where each gate is a small variational circuit."""
    class QGate(tq.QuantumModule):
        def __init__(self, n_wires: int, n_ops: int = 20):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.random = tq.RandomLayer(n_ops=n_ops, wires=range(n_wires))
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.cnot = tq.CNOT
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, qdev: tq.QuantumDevice):
            self.random(qdev)
            self.rz(qdev, wires=0)
            self.cnot(qdev, wires=[0, 1])
            self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 4, shift: float = np.pi/2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.shift = shift

        self.forget_gate = self.QGate(n_qubits)
        self.input_gate = self.QGate(n_qubits)
        self.update_gate = self.QGate(n_qubits)
        self.output_gate = self.QGate(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, inputs: torch.Tensor, states: tuple | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = self._quantum_gate(self.linear_forget(combined), self.forget_gate)
            i = self._quantum_gate(self.linear_input(combined), self.input_gate)
            g = torch.tanh(self._quantum_gate(self.linear_update(combined), self.update_gate))
            o = self._quantum_gate(self.linear_output(combined), self.output_gate)
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        logits = self.classifier(stacked)
        probs = torch.sigmoid(logits + self.shift)
        return torch.cat((probs, 1 - probs), dim=-1), (hx, cx)

    def _quantum_gate(self, params: torch.Tensor, gate: tq.QuantumModule) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=params.shape[0], device=params.device)
        gate.encoder(qdev, params)
        gate(qdev)
        out = gate.measure(qdev)
        return out.mean(dim=1, keepdim=True)

    def _init_states(self, inputs: torch.Tensor, states: tuple | None):
        if states is not None:
            return states
        batch = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch, self.hidden_dim, device=device), torch.zeros(batch, self.hidden_dim, device=device)

__all__ = ["QLSTMGen384"]
