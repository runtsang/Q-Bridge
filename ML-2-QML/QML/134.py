import torch
import torch.nn as nn
import pennylane as qml
from typing import Tuple

class QGate(nn.Module):
    """Variational quantum circuit that maps an input vector to a probability vector."""
    def __init__(
        self,
        in_features: int,
        n_qubits: int,
        n_layers: int = 2,
        device: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, n_qubits)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.wires = list(range(n_qubits))
        self.params = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)

        self.qnode = qml.QNode(
            self._circuit,
            device=qml.device(device, wires=self.wires),
            interface="torch",
            diff_method="backprop",
        )

    def _circuit(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        # x: [batch, n_qubits]
        for i, wire in enumerate(self.wires):
            qml.RX(x[:, i], wires=wire)
        for layer in range(self.n_layers):
            for wire in self.wires:
                qml.Rot(
                    params[layer, wire, 0],
                    params[layer, wire, 1],
                    params[layer, wire, 2],
                    wires=wire,
                )
        for i in range(len(self.wires) - 1):
            qml.CNOT(self.wires[i], self.wires[i + 1])
        return [qml.expval(qml.PauliZ(wire)) for wire in self.wires]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_lin = self.linear(x)
        return self.qnode(x_lin, self.params)

class QLSTM(nn.Module):
    """Hybrid LSTM where each gate is realised by a variational quantum circuit."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        n_layers: int = 2,
        device: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = device

        self.forget_gate = QGate(input_dim + hidden_dim, n_qubits, n_layers, device)
        self.input_gate = QGate(input_dim + hidden_dim, n_qubits, n_layers, device)
        self.update_gate = QGate(input_dim + hidden_dim, n_qubits, n_layers, device)
        self.output_gate = QGate(input_dim + hidden_dim, n_qubits, n_layers, device)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
            g = torch.tanh(self.update_gate(self.linear_update(combined)))
            o = torch.sigmoid(self.output_gate(self.linear_output(combined)))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )
