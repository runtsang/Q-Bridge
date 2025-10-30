import torch
import torch.nn as nn
import torch.nn.functional as F

class QLSTM(nn.Module):
    """
    Classical LSTM cell with a lightweight variational circuit for the gate activations.
    The circuit depth is controlled by ``depth`` and a depth‑penalty term can be added to the loss
    to encourage shallow quantum emulations.
    """
    class QLayer(nn.Module):
        """
        Parameter‑efficient variational layer that simulates a small quantum circuit.
        It consists of ``depth`` linear layers followed by a final readout.
        """
        def __init__(self, n_qubits: int, depth: int = 1):
            super().__init__()
            self.n_qubits = n_qubits
            self.depth = depth
            self.layers = nn.ModuleList(
                [nn.Linear(n_qubits, n_qubits) for _ in range(depth)]
            )
            self.readout = nn.Linear(n_qubits, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for layer in self.layers:
                x = torch.tanh(layer(x))
            return self.readout(x).squeeze(-1)

    def __init__(self, input_dim: int, hidden_dim: int,
                 n_qubits: int, depth: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth

        self.forget_gate = self.QLayer(n_qubits, depth)
        self.input_gate = self.QLayer(n_qubits, depth)
        self.update_gate = self.QLayer(n_qubits, depth)
        self.output_gate = self.QLayer(n_qubits, depth)

        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits)

        self.depth_penalty = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float32), requires_grad=False
        )

    def forward(self, inputs: torch.Tensor,
                states: tuple | None = None) -> tuple:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.forget_lin(combined)))
            i = torch.sigmoid(self.input_gate(self.input_lin(combined)))
            g = torch.tanh(self.update_gate(self.update_lin(combined)))
            o = torch.sigmoid(self.output_gate(self.output_lin(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self, inputs: torch.Tensor,
                     states: tuple | None = None) -> tuple:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between the classical
    LSTM and the hybrid quantum‑classical LSTM.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int,
                 vocab_size: int, tagset_size: int,
                 n_qubits: int = 0, depth: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim,
                              n_qubits=n_qubits, depth=depth)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)
