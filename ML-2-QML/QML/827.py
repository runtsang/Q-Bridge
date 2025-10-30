import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from typing import Tuple, Optional

class QLSTMEnhanced(nn.Module):
    """
    Hybrid LSTM cell with a quantum forget gate implemented via a PennyLane circuit.
    The other gates are classical linear layers.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_qubits: int = 0,
                 skip: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.skip = skip
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Classical linear projections for all gates
        self.linear_f = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.linear_i = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.linear_o = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.linear_g = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Quantum forget gate
        if not skip and n_qubits > 0:
            self.dev = qml.device("default.qubit", wires=n_qubits)
            self.quantum_weights = nn.Parameter(torch.randn(n_qubits))
            def circuit(inputs, weights):
                # Encode inputs as RX rotations
                for i in range(n_qubits):
                    qml.RX(inputs[i], wires=i)
                # Variational layer
                for i in range(n_qubits):
                    qml.RY(weights[i], wires=i)
                # Entanglement
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                return qml.expval(qml.PauliZ(0))
            self.quantum_forget = qml.QNode(circuit, self.dev, interface="torch")
        else:
            self.quantum_forget = None

    def _quantum_forget_gate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the forget gate using the quantum circuit.
        """
        batch_size = x.shape[0]
        outputs = []
        for i in range(batch_size):
            inp = x[i, :self.n_qubits]
            out = self.quantum_forget(inp, self.quantum_weights)
            outputs.append(out)
        return torch.sigmoid(torch.stack(outputs))

    def forward(self,
                inputs: torch.Tensor,
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            if self.quantum_forget is not None:
                f = self._quantum_forget_gate(combined)
            else:
                f = torch.sigmoid(self.linear_f(combined))
            i = torch.sigmoid(self.linear_i(combined))
            g = torch.tanh(self.linear_g(combined))
            o = torch.sigmoid(self.linear_o(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            hx = self.dropout(hx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can use QLSTMEnhanced or nn.LSTM.
    Supports optional auxiliary task and dropout regularisation.
    """
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 aux_tagset_size: Optional[int] = None,
                 n_qubits: int = 0,
                 skip: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTMEnhanced(embedding_dim,
                                      hidden_dim,
                                      n_qubits=n_qubits,
                                      skip=skip,
                                      dropout=dropout)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.aux_tagset_size = aux_tagset_size
        if aux_tagset_size is not None:
            self.hidden2aux = nn.Linear(hidden_dim, aux_tagset_size)

    def forward(self, sentence: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        embeds = self.word_embeddings(sentence)
        if isinstance(self.lstm, nn.LSTM):
            lstm_out, _ = self.lstm(embeds)
        else:
            lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        tag_logits = self.hidden2tag(lstm_out)
        tag_logits = F.log_softmax(tag_logits, dim=-1)
        aux_logits = None
        if self.aux_tagset_size is not None:
            aux_logits = self.hidden2aux(lstm_out)
            aux_logits = F.log_softmax(aux_logits, dim=-1)
        return tag_logits, aux_logits

__all__ = ["QLSTMEnhanced", "LSTMTagger"]
