import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

class QLSTMQuantum(nn.Module):
    """
    Quantum‑augmented LSTM cell implemented with PennyLane.
    Each gate is a variational circuit with depth `q_depth`.
    The circuit shares parameters across all time steps.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_qubits: int,
                 q_depth: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.q_depth = q_depth

        # Classical linear maps
        self.forget_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_lin  = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Quantum device and circuit templates
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.q_gate = self._build_qgate()

    def _build_qgate(self):
        def circuit(inputs, params):
            # Encode inputs into qubits via RX rotations
            for i in range(self.n_qubits):
                qml.RX(inputs[i], wires=i)
            # Variational layers
            for d in range(self.q_depth):
                for i in range(self.n_qubits):
                    qml.RZ(params[d, i], wires=i)
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
            # Measurement in Z basis
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        return circuit

    def _qnode(self, inputs, params):
        @qml.qnode(self.dev, interface="torch")
        def circuit():
            return self.q_gate(inputs, params)
        return circuit()

    def forward(self,
                inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []

        # Shared trainable parameters for each gate
        gate_params = {
            "f": nn.Parameter(torch.randn(self.q_depth, self.n_qubits)),
            "i": nn.Parameter(torch.randn(self.q_depth, self.n_qubits)),
            "g": nn.Parameter(torch.randn(self.q_depth, self.n_qubits)),
            "o": nn.Parameter(torch.randn(self.q_depth, self.n_qubits)),
        }

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            f = torch.sigmoid(self.forget_lin(combined))
            i = torch.sigmoid(self.input_lin(combined))
            g = torch.tanh(self.update_lin(combined))
            o = torch.sigmoid(self.output_lin(combined))

            # Quantum augmentation
            f_q = torch.sigmoid(self._qnode(f, gate_params["f"]))
            i_q = torch.sigmoid(self._qnode(i, gate_params["i"]))
            g_q = torch.tanh(self._qnode(g, gate_params["g"]))
            o_q = torch.sigmoid(self._qnode(o, gate_params["o"]))

            cx = f_q * cx + i_q * g_q
            hx = o_q * torch.tanh(cx)

            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class LSTMTaggerQuantum(nn.Module):
    """
    Tagger that uses the PennyLane‑based quantum LSTM when `n_qubits > 0`,
    otherwise falls back to the classical LSTM.
    """
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 q_depth: int = 2):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTMQuantum(embedding_dim,
                                     hidden_dim,
                                     n_qubits=n_qubits,
                                     q_depth=q_depth)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTMQuantum", "LSTMTaggerQuantum"]
