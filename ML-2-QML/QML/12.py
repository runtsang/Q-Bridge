import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QLSTMPlus(nn.Module):
    """
    Quantum‑enhanced LSTM where each gate is realised by a variational circuit.
    Supports the same API as the classical version, enabling a drop‑in swap.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        dropout: float = 0.0,
        use_layernorm: bool = False,
        use_residual: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.use_layernorm = use_layernorm
        self.use_residual = use_residual

        # Quantum device
        self.device = qml.device("default.qubit", wires=n_qubits)

        # Variational ansatz parameters (one per gate)
        num_layers = 2
        self.forget_weights = nn.Parameter(torch.randn(num_layers, n_qubits, 3))
        self.input_weights = nn.Parameter(torch.randn(num_layers, n_qubits, 3))
        self.update_weights = nn.Parameter(torch.randn(num_layers, n_qubits, 3))
        self.output_weights = nn.Parameter(torch.randn(num_layers, n_qubits, 3))

        # Helper QNode
        def _gate(inputs, weights):
            # Encode the classical input into rotation angles
            for i, val in enumerate(inputs):
                qml.RX(val, wires=i)
            qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.quantum_gate = qml.qnode(self.device, interface="torch")(_gate)

        # Classical post‑processing of the quantum measurement
        self.forget_linear = nn.Linear(n_qubits, hidden_dim)
        self.input_linear = nn.Linear(n_qubits, hidden_dim)
        self.update_linear = nn.Linear(n_qubits, hidden_dim)
        self.output_linear = nn.Linear(n_qubits, hidden_dim)

        if self.use_layernorm:
            self.ln_forget = nn.LayerNorm(hidden_dim)
            self.ln_input = nn.LayerNorm(hidden_dim)
            self.ln_update = nn.LayerNorm(hidden_dim)
            self.ln_output = nn.LayerNorm(hidden_dim)

        # Attention read‑out
        self.attn = nn.Linear(hidden_dim, 1)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        inputs: (seq_len, batch, input_dim)
        returns: (seq_len, batch, hidden_dim), (hx, cx)
        """
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            # Quantum gates
            # Process each sample in the batch separately to keep the example simple
            f_q = []
            i_q = []
            g_q = []
            o_q = []
            for sample in combined:
                qout = self.quantum_gate(sample, self.forget_weights)
                f_q.append(qout)
                qout = self.quantum_gate(sample, self.input_weights)
                i_q.append(qout)
                qout = self.quantum_gate(sample, self.update_weights)
                g_q.append(qout)
                qout = self.quantum_gate(sample, self.output_weights)
                o_q.append(qout)
            f_q = torch.stack(f_q, dim=0)
            i_q = torch.stack(i_q, dim=0)
            g_q = torch.stack(g_q, dim=0)
            o_q = torch.stack(o_q, dim=0)

            # Map quantum outputs to hidden dimension
            f = torch.sigmoid(self.forget_linear(f_q))
            i = torch.sigmoid(self.input_linear(i_q))
            g = torch.tanh(self.update_linear(g_q))
            o = torch.sigmoid(self.output_linear(o_q))

            if self.use_layernorm:
                f = self.ln_forget(f)
                i = self.ln_input(i)
                g = self.ln_update(g)
                o = self.ln_output(o)

            f, i, g, o = map(self.dropout, (f, i, g, o))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)

            if self.use_residual:
                hx = hx + combined[:, :self.hidden_dim]

            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)

        # Gated‑attention read‑out over the whole sequence
        attn_weights = F.softmax(self.attn(outputs).squeeze(-1), dim=0)  # (seq_len,)
        context = torch.sum(outputs * attn_weights.unsqueeze(-1), dim=0)  # (batch, hidden_dim)
        # Context is not returned to preserve API

        return outputs, (hx, cx)

__all__ = ["QLSTMPlus"]
