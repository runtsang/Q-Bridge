import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import uniform_
import torchquantum as tq
import torchquantum.functional as tqf


class QLSTMGen(nn.Module):
    """Hybrid LSTM where each gate is realised by a variational quantum circuit.
    The hidden state is represented classically, but the gate tensors are
    produced from expectation values of Pauli‑Z after a small quantum circuit.
    """

    class QGate(tq.QuantumModule):
        """A small quantum circuit that outputs a vector of size n_qubits."""
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            # Variational parameters: RX, RY, RZ for each qubit
            self.params = nn.Parameter(torch.randn(n_qubits * 3))
            # Simple entangling chain
            self.cnot_chain = [tq.CNOT() for _ in range(n_qubits - 1)]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x : (batch, n_qubits) – input features encoded as rotation angles
            qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=x.shape[0], device=x.device)
            # Encode input as rotations
            for i in range(self.n_qubits):
                tqf.rx(qdev, wires=i, params=x[:, i])
            # Apply variational parameters
            for i, theta in enumerate(self.params):
                if i % 3 == 0:
                    tqf.rx(qdev, wires=i // 3, params=theta)
                elif i % 3 == 1:
                    tqf.ry(qdev, wires=i // 3, params=theta)
                else:
                    tqf.rz(qdev, wires=i // 3, params=theta)
            # Entangle
            for i, cnot in enumerate(self.cnot_chain):
                cnot(qdev, wires=[i, i + 1])
            # Expectation of Pauli‑Z for each qubit
            return tqf.expectation(qdev, tq.PauliZ)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Quantum gates
        self.forget_gate = self.QGate(n_qubits)
        self.input_gate = self.QGate(n_qubits)
        self.update_gate = self.QGate(n_qubits)
        self.output_gate = self.QGate(n_qubits)

        # Classical linear layers that feed the quantum circuits
        self.forget_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_linear = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Map quantum outputs to hidden dimension
        self.q_to_hidden = nn.Linear(n_qubits, hidden_dim)

    def forward(self, inputs: torch.Tensor, states: tuple = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.forget_linear(combined)))
            i = torch.sigmoid(self.input_gate(self.input_linear(combined)))
            g = torch.tanh(self.update_gate(self.update_linear(combined)))
            o = torch.sigmoid(self.output_gate(self.output_linear(combined)))

            # Map quantum outputs to hidden dimension
            f = self.q_to_hidden(f)
            i = self.q_to_hidden(i)
            g = self.q_to_hidden(g)
            o = self.q_to_hidden(o)

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self, inputs, states):
        if states is not None:
            return states
        batch = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch, self.hidden_dim, device=device), torch.zeros(batch, self.hidden_dim, device=device)

__all__ = ["QLSTMGen"]
