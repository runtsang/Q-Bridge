import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple, Optional

class QLSTMHybrid(nn.Module):
    """
    Quantum‑enhanced LSTM with the same API as the classical variant.
    Each gate is a depth‑controlled variational circuit that outputs a real‑valued
    vector of size `n_qubits`.  The vector is linearly mapped to the hidden
    dimension.  Supports dynamic masking, a lightweight attention head, and
    a coherence regularisation term in the loss.
    """

    class QLayer(tq.QuantumModule):
        """
        Encodes the input into a quantum state, applies a parameterised circuit,
        and measures all qubits in the Z basis.
        """
        def __init__(self, n_wires: int, depth: int = 1):
            super().__init__()
            self.n_wires = n_wires
            self.depth = depth
            # Parameterised circuit: depth layers of RX rotations
            self.params = nn.ParameterList()
            for _ in range(depth):
                for _ in range(n_wires):
                    self.params.append(nn.Parameter(torch.randn(1)))
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            x: (batch, n_wires) real values
            """
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            # Encode real input via RX rotations
            for w in range(self.n_wires):
                tqf.rx(qdev, x[:, w], wires=[w])
            # Parameterised layers
            idx = 0
            for _ in range(self.depth):
                for w in range(self.n_wires):
                    tqf.rx(qdev, self.params[idx], wires=[w])
                    idx += 1
                # Entangling CNOT chain
                for w in range(self.n_wires - 1):
                    tqf.cnot(qdev, wires=[w, w + 1])
            return self.measure(qdev)

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_qubits: int,
                 depth: int = 1,
                 attention_dim: Optional[int] = None,
                 device: torch.device | None = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth
        self.attention_dim = attention_dim

        if device is None:
            device = torch.device('cpu')
        self.device = device

        # Quantum layers for each gate
        self.forget_q = self.QLayer(n_qubits, depth)
        self.input_q = self.QLayer(n_qubits, depth)
        self.update_q = self.QLayer(n_qubits, depth)
        self.output_q = self.QLayer(n_qubits, depth)

        # Linear projections from quantum outputs to hidden dimension
        self.forget_lin = nn.Linear(n_qubits, hidden_dim)
        self.input_lin = nn.Linear(n_qubits, hidden_dim)
        self.update_lin = nn.Linear(n_qubits, hidden_dim)
        self.output_lin = nn.Linear(n_qubits, hidden_dim)

        # Optional attention head
        if attention_dim is not None:
            self.attention = nn.Linear(hidden_dim, attention_dim)
            self.attention_context = nn.Linear(attention_dim, hidden_dim)
        else:
            self.attention = None

    def forward(self,
                inputs: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        inputs: (seq_len, batch, input_dim)
        mask: (seq_len, batch) with 1 for valid tokens, 0 for padding
        states: (h_0, c_0) each (batch, hidden_dim)
        """
        hx, cx = self._init_states(inputs, states)
        seq_len, batch_size, _ = inputs.size()
        outputs = []

        for t in range(seq_len):
            x_t = inputs[t]
            combined = torch.cat([x_t, hx], dim=1)

            # Linear projection to quantum input size
            q_in = combined  # (batch, input_dim + hidden_dim)
            # Use first n_qubits dimensions for quantum encoding
            q_in_reduced = q_in[:, :self.n_qubits]  # (batch, n_qubits)

            f_q = self.forget_q(q_in_reduced)
            i_q = self.input_q(q_in_reduced)
            g_q = self.update_q(q_in_reduced)
            o_q = self.output_q(q_in_reduced)

            f = torch.sigmoid(self.forget_lin(f_q))
            i = torch.sigmoid(self.input_lin(i_q))
            g = torch.tanh(self.update_lin(g_q))
            o = torch.sigmoid(self.output_lin(o_q))

            if mask is not None:
                mask_t = mask[t].unsqueeze(1).to(dtype=f.dtype)
                f = f * mask_t + (1 - mask_t) * 1.0
                i = i * mask_t + (1 - mask_t) * 0.0
                g = g * mask_t + (1 - mask_t) * 0.0
                o = o * mask_t + (1 - mask_t) * 1.0

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)

            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)  # (seq_len, batch, hidden_dim)

        # Optional attention over hidden states
        if self.attention is not None:
            attn_weights = F.softmax(self.attention(outputs), dim=0)
            context = torch.sum(attn_weights * outputs, dim=0)
            outputs = context.unsqueeze(0).expand(seq_len, batch_size, -1)

        return outputs, (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

    def compute_loss(self,
                     logits: torch.Tensor,
                     targets: torch.Tensor,
                     mask: Optional[torch.Tensor] = None,
                     lambda_coherence: float = 0.0) -> torch.Tensor:
        """
        Negative log likelihood loss plus optional quantum coherence regularisation.
        """
        loss = F.nll_loss(logits, targets, reduction='none')
        if mask is not None:
            loss = loss * mask
        loss = loss.mean()

        if lambda_coherence > 0.0:
            # Simple coherence penalty: mean absolute value of product of distinct qubit outputs
            # Here we approximate by taking the first two qubit outputs from the last quantum layer.
            q_out = self.output_q(torch.zeros_like(logits[:, :self.n_qubits]))
            coherence = torch.abs(torch.mean(q_out[:, 0] * q_out[:, 1:]))
            loss += lambda_coherence * coherence

        return loss
