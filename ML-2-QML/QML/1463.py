import torch
import torch.nn as nn
import pennylane as qml

class _QLayerQuantum(nn.Module):
    """Quantum gate implemented as a parameterised Pennylane circuit.
    The circuit measures the Pauli‑Z expectation of each qubit, producing
    a differentiable vector that can be fed into the classical LSTM gates."""
    def __init__(self, n_qubits: int, backend: str = 'default.qubit'):
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device(backend, wires=n_qubits)

        # Trainable parameters: one (RX, RY, RZ) set per qubit
        self.params = nn.Parameter(torch.randn(n_qubits, 3))

        @qml.qnode(self.dev, interface='torch')
        def circuit(inputs: torch.Tensor, params: torch.Tensor):
            # Encode the gate activation into RX rotations
            for i in range(n_qubits):
                qml.RX(inputs[i], wires=i)
            # Apply a small parameterised block per qubit
            for i in range(n_qubits):
                qml.RY(params[i, 0], wires=i)
                qml.RZ(params[i, 1], wires=i)
                # Small entanglement pattern
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
            # Measure Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape (batch, n_qubits)
        return self.circuit(x, self.params)

class QLSTMHybrid(nn.Module):
    """Hybrid LSTM that uses Pennylane quantum gates when ``n_qubits > 0``.
    The interface is identical to the classical version, allowing
    seamless switching between regimes."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0,
                 dropout: float = 0.0, use_layernorm: bool = False,
                 backend: str = 'default.qubit'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.dropout = nn.Dropout(dropout)
        self.use_layernorm = use_layernorm

        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        if n_qubits > 0:
            self.forget_q = _QLayerQuantum(n_qubits, backend=backend)
            self.input_q = _QLayerQuantum(n_qubits, backend=backend)
            self.update_q = _QLayerQuantum(n_qubits, backend=backend)
            self.output_q = _QLayerQuantum(n_qubits, backend=backend)
            self.quantum_to_hidden = nn.Linear(n_qubits, hidden_dim)
        else:
            self.forget_q = self.input_q = self.update_q = self.output_q = None

        if use_layernorm:
            self.ln_f = nn.LayerNorm(hidden_dim)
            self.ln_i = nn.LayerNorm(hidden_dim)
            self.ln_u = nn.LayerNorm(hidden_dim)
            self.ln_o = nn.LayerNorm(hidden_dim)

    def _init_states(self, inputs: torch.Tensor,
                     states: tuple[torch.Tensor, torch.Tensor] | None = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(self, inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))

            if self.n_qubits > 0:
                f_q = self.forget_q(f)
                i_q = self.input_q(i)
                g_q = self.update_q(g)
                o_q = self.output_q(o)

                f = torch.sigmoid(self.quantum_to_hidden(f_q))
                i = torch.sigmoid(self.quantum_to_hidden(i_q))
                g = torch.tanh(self.quantum_to_hidden(g_q))
                o = torch.sigmoid(self.quantum_to_hidden(o_q))

            if self.use_layernorm:
                f = self.ln_f(f)
                i = self.ln_i(i)
                g = self.ln_u(g)
                o = self.ln_o(o)

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            hx = self.dropout(hx)

            outputs.append(hx.unsqueeze(0))

        output = torch.cat(outputs, dim=0)
        return output, (hx, cx)

__all__ = ['QLSTMHybrid']
