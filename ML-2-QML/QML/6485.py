import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from typing import Tuple, Optional

class QLSTMGen313(nn.Module):
    """
    Quantum‑augmented LSTM cell. Each gate is multiplied by a scaling factor
    produced by a variational quantum circuit. The circuit is a simple
    two‑layer ansatz consisting of rotations (parameterised by the gate
    activation) followed by a trainable rotation layer and a chain of CNOTs.
    """
    def __init__(self, input_dim: int, hidden_dim: int,
                 n_qubits: int, depth: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth

        self.forget_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Quantum device and trainable parameters
        self.dev = qml.device("default.qubit", wires=self.n_qubits, shots=None, interface="torch")
        self.scale_params = nn.ParameterList([nn.Parameter(torch.randn(self.n_qubits))
                                              for _ in range(4)])

        # Create a separate QNode for each gate
        self.scale_qnode_f = self._make_qnode(0)
        self.scale_qnode_i = self._make_qnode(1)
        self.scale_qnode_g = self._make_qnode(2)
        self.scale_qnode_o = self._make_qnode(3)

    def _make_qnode(self, gate_idx: int):
        params = self.scale_params[gate_idx]

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def qnode(x):
            # x is (batch, n_qubits)
            for w in range(self.n_qubits):
                qml.RY(x[:, w], wires=w)
            for w in range(self.n_qubits):
                qml.RY(params[w], wires=w)
            # entangling layer
            for w in range(self.n_qubits - 1):
                qml.CNOT(wires=[w, w + 1])
            # expectation value on first qubit
            return qml.expval(qml.PauliZ(0))
        return qnode

    def _quantum_scale(self, x: torch.Tensor, qnode) -> torch.Tensor:
        # x shape (batch, hidden_dim); use first n_qubits features
        x_slice = x[:, :self.n_qubits]
        return torch.sigmoid(qnode(x_slice))

    def forward(self, inputs: torch.Tensor,
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_lin(combined))
            i = torch.sigmoid(self.input_lin(combined))
            g = torch.tanh(self.update_lin(combined))
            o = torch.sigmoid(self.output_lin(combined))

            scale_f = self._quantum_scale(f, self.scale_qnode_f).unsqueeze(-1)
            scale_i = self._quantum_scale(i, self.scale_qnode_i).unsqueeze(-1)
            scale_g = self._quantum_scale(g, self.scale_qnode_g).unsqueeze(-1)
            scale_o = self._quantum_scale(o, self.scale_qnode_o).unsqueeze(-1)

            f = f * scale_f
            i = i * scale_i
            g = g * scale_g
            o = o * scale_o

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self, inputs: torch.Tensor,
                     states: Optional[Tuple[torch.Tensor, torch.Tensor]]
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class LSTMTaggerGen313(nn.Module):
    """
    Sequence tagging model that uses the quantum‑augmented LSTM.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int,
                 vocab_size: int, tagset_size: int,
                 n_qubits: int, depth: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTMGen313(embedding_dim, hidden_dim,
                                n_qubits=n_qubits, depth=depth)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTMGen313", "LSTMTaggerGen313"]
