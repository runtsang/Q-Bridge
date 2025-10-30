import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import pennylane as qml

class QGate(nn.Module):
    """
    Variational quantum gate that maps a vector of length n_qubits to a vector
    of the same length. The circuit depth can be increased for expressivity.
    """
    def __init__(self, n_qubits: int, depth: int = 1):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        # Parameters for each qubit in each layer: rotation angles (rx, ry, rz)
        self.params = nn.Parameter(torch.randn(depth, n_qubits, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (batch, n_qubits)
        """
        dev = qml.device("default.qubit", wires=self.n_qubits, shots=None)
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, params):
            # Encode inputs with RX rotations
            for i in range(self.n_qubits):
                qml.RX(inputs[:, i], wires=i)
            # Variational layers
            for l in range(self.depth):
                for q in range(self.n_qubits):
                    qml.RX(params[l, q, 0], wires=q)
                    qml.RY(params[l, q, 1], wires=q)
                    qml.RZ(params[l, q, 2], wires=q)
                # Entangle qubits in a ring
                for q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])
            # Measure Z on each qubit
            return torch.stack([qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits)], dim=1)
        return circuit(x, self.params)

class QLSTMGen(nn.Module):
    """
    Quantum‑enhanced LSTM with optional depth and hybrid gating.
    Parameters
    ----------
    input_dim : int
        Dimensionality of input embeddings.
    hidden_dim : int
        Size of hidden state. Must equal n_qubits for quantum gates.
    n_qubits : int, default 0
        If 0, the model behaves as a classical LSTM.
    depth : int, default 1
        Depth of each quantum gate (number of variational layers).
    use_hybrid : bool, default True
        If True, each gate output is a weighted sum of classical linear
        output and quantum measurement, enabling a smooth interpolation.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0,
                 depth: int = 1, use_hybrid: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth
        self.use_hybrid = use_hybrid

        # Classical linear projections
        self.forget_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Quantum gates
        if n_qubits > 0:
            self.forget_q = QGate(n_qubits, depth)
            self.input_q = QGate(n_qubits, depth)
            self.update_q = QGate(n_qubits, depth)
            self.output_q = QGate(n_qubits, depth)
            # Hybrid weighting parameter per gate
            self.alpha = nn.Parameter(torch.full((4, hidden_dim), 0.5))
        else:
            self.forget_q = self.input_q = self.update_q = self.output_q = None

    def forward(self, inputs: torch.Tensor,
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        inputs: Tensor of shape (batch, seq_len, input_dim)
        """
        if states is None:
            h0 = torch.zeros(self.hidden_dim, inputs.size(0), device=inputs.device)
            c0 = torch.zeros(self.hidden_dim, inputs.size(0), device=inputs.device)
        else:
            h0, c0 = states

        hx, cx = h0, c0
        outputs = []

        for x in inputs.unbind(dim=1):
            combined = torch.cat([x, hx], dim=1)

            f_lin = torch.sigmoid(self.forget_lin(combined))
            i_lin = torch.sigmoid(self.input_lin(combined))
            g_lin = torch.tanh(self.update_lin(combined))
            o_lin = torch.sigmoid(self.output_lin(combined))

            if self.n_qubits > 0:
                f_q = torch.sigmoid(self.forget_q(f_lin))
                i_q = torch.sigmoid(self.input_q(i_lin))
                g_q = torch.tanh(self.update_q(g_lin))
                o_q = torch.sigmoid(self.output_q(o_lin))

                if self.use_hybrid:
                    # weighted sum: alpha * linear + (1-alpha) * quantum
                    alpha_f, alpha_i, alpha_g, alpha_o = self.alpha
                    f = alpha_f * f_lin + (1 - alpha_f) * f_q
                    i = alpha_i * i_lin + (1 - alpha_i) * i_q
                    g = alpha_g * g_lin + (1 - alpha_g) * g_q
                    o = alpha_o * o_lin + (1 - alpha_o) * o_q
                else:
                    f, i, g, o = f_q, i_q, g_q, o_q
            else:
                f, i, g, o = f_lin, i_lin, g_lin, o_lin

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)  # (batch, seq_len, hidden_dim)
        return outputs, (hx, cx)

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between classical and quantum LSTM.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 0, depth: int = 1,
                 use_hybrid: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTMGen(embedding_dim, hidden_dim,
                             n_qubits=n_qubits, depth=depth,
                             use_hybrid=use_hybrid)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.LongTensor) -> torch.Tensor:
        """
        sentence: LongTensor of shape (batch, seq_len)
        """
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=-1)

__all__ = ["QLSTMGen", "LSTMTagger"]
