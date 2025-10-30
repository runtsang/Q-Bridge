import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import pennylane as qml


class QLSTMHybrid(nn.Module):
    """
    Quantum LSTM cell with variational gates.
    """

    class QLayer(nn.Module):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            self.dev = qml.device("default.qubit", wires=n_qubits)

            @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
            def circuit(x):
                # Encode each element of x onto a qubit
                for i in range(n_qubits):
                    qml.RY(x[:, i], wires=i)
                # Entangling layer
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
                # Measure expectation values
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

            self.qnode = circuit

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, n_qubits)
            return self.qnode(x)

    def __init__(self, input_dim: int, hidden_dim: int,
                 n_qubits: int = 8,
                 dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.dropout = nn.Dropout(dropout)

        # Encoder: 1D conv as in classical version
        self.encoder = nn.Conv1d(in_channels=input_dim,
                                 out_channels=hidden_dim,
                                 kernel_size=3,
                                 padding=1,
                                 bias=False)

        self.layer_norm = nn.LayerNorm(2 * hidden_dim)

        # Quantum gates
        self.forget_q = self.QLayer(n_qubits)
        self.input_q = self.QLayer(n_qubits)
        self.update_q = self.QLayer(n_qubits)
        self.output_q = self.QLayer(n_qubits)

        self.forget_lin = nn.Linear(2 * hidden_dim, n_qubits)
        self.input_lin = nn.Linear(2 * hidden_dim, n_qubits)
        self.update_lin = nn.Linear(2 * hidden_dim, n_qubits)
        self.output_lin = nn.Linear(2 * hidden_dim, n_qubits)

    def forward(self,
                inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)

        # Encode inputs
        seq_len, batch_size, _ = inputs.shape
        x = inputs.permute(1, 2, 0)  # (batch, input_dim, seq_len)
        encoded = self.encoder(x)   # (batch, hidden_dim, seq_len)
        encoded = encoded.permute(2, 0, 1)  # (seq_len, batch, hidden_dim)

        outputs = []
        for x_t in encoded.unbind(dim=0):
            combined = torch.cat([x_t, hx], dim=1)
            combined_ln = self.layer_norm(combined)

            # Linear projection to qubit dimension
            f_q = self.forget_q(self.forget_lin(combined_ln))
            i_q = self.input_q(self.input_lin(combined_ln))
            g_q = self.update_q(self.update_lin(combined_ln))
            o_q = self.output_q(self.output_lin(combined_ln))

            # Apply activation functions
            f = torch.sigmoid(f_q)
            i = torch.sigmoid(i_q)
            g = torch.tanh(g_q)
            o = torch.sigmoid(o_q)

            f = self.dropout(f)
            i = self.dropout(i)
            g = self.dropout(g)
            o = self.dropout(o)

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Tuple[torch.Tensor, torch.Tensor] | None
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


class LSTMTagger(nn.Module):
    """
    Sequence tagging model using the quantum hybrid LSTM.
    """

    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 8,
                 dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTMHybrid(embedding_dim,
                                hidden_dim,
                                n_qubits=n_qubits,
                                dropout=dropout)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTMHybrid", "LSTMTagger"]
