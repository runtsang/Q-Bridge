"""
Quantum modules: self‑attention circuit and quantum‑enhanced LSTM.
"""

import pennylane as qml
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumSelfAttention(nn.Module):
    """Variational quantum circuit implementing self‑attention."""
    def __init__(self, n_qubits: int = 4, device: str = "default.qubit"):
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device(device, wires=n_qubits)

    def run(self, rotation_params: torch.Tensor, entangle_params: torch.Tensor,
            inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rotation_params: Tensor of shape (n_qubits*3,)
            entangle_params: Tensor of shape (n_qubits-1,)
            inputs: Tensor of shape (seq_len, batch, 2)
        Returns:
            Attention weights tensor of shape (seq_len, batch, 1)
        """
        seq_len, batch, _ = inputs.shape
        flat_inputs = inputs.view(seq_len * batch, -1)

        @qml.qnode(self.dev, interface="torch")
        def circuit(x, rot, ent):
            # Input encoding
            for i, val in enumerate(x):
                qml.RX(val, wires=i)
            # Rotation layer
            for i in range(self.n_qubits):
                qml.RX(rot[3 * i], wires=i)
                qml.RY(rot[3 * i + 1], wires=i)
                qml.RZ(rot[3 * i + 2], wires=i)
            # Entangling layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(ent[i], wires=i)
            # Measurement: expectation of PauliZ on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        results = circuit(flat_inputs, rotation_params, entangle_params)
        attn = torch.stack(results, dim=0)          # (seq_len*batch, n_qubits)
        attn = torch.softmax(attn, dim=1)
        weights = attn.mean(dim=1).unsqueeze(-1)    # (seq_len*batch, 1)
        return weights.view(seq_len, batch, 1)

class QLSTM(nn.Module):
    """LSTM cell where each gate is realized by a small quantum circuit."""
    class QGate(tq.QuantumModule):
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

        self.forget_gate = self.QGate(n_qubits)
        self.input_gate = self.QGate(n_qubits)
        self.update_gate = self.QGate(n_qubits)
        self.output_gate = self.QGate(n_qubits)

        self.forget_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_linear = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.forget_linear(combined)))
            i = torch.sigmoid(self.input_gate(self.input_linear(combined)))
            g = torch.tanh(self.update_gate(self.update_linear(combined)))
            o = torch.sigmoid(self.output_gate(self.output_linear(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self, inputs: torch.Tensor,
                     states: Tuple[torch.Tensor, torch.Tensor] | None
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device)
        )
