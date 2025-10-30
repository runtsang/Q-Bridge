import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import pennylane as qml

class QuantumGate(nn.Module):
    """Quantum gate implemented as a PennyLane qnode."""
    def __init__(self, n_qubits: int, depth: int = 1, device: str = "default.qubit"):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.dev = qml.device(device, wires=n_qubits, shots=1024)
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs):
            # Encode inputs into rotation angles
            for i in range(n_qubits):
                qml.RX(inputs[i], wires=i)
            # Apply layers
            for d in range(depth):
                for i in range(n_qubits):
                    qml.RZ(0.1, wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i+1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (batch, n_qubits)
        return self.circuit(x)

class UnifiedQLSTMCell(nn.Module):
    """Hybrid LSTM cell that mixes classical linear gates with quantum gates."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0,
                 depth: int = 1, mix_factor: float = 0.5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.mix_factor = mix_factor
        self.depth = depth

        # Linear projections for classical gates (hidden_dim outputs)
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        if n_qubits > 0:
            # Linear projections for quantum gates (n_qubits outputs)
            self.forget_q_proj = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.input_q_proj = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.update_q_proj = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.output_q_proj = nn.Linear(input_dim + hidden_dim, n_qubits)
            # Quantum gates
            self.forget_qc = QuantumGate(n_qubits, depth)
            self.input_qc = QuantumGate(n_qubits, depth)
            self.update_qc = QuantumGate(n_qubits, depth)
            self.output_qc = QuantumGate(n_qubits, depth)
        else:
            self.forget_qc = None
            self.input_qc = None
            self.update_qc = None
            self.output_qc = None

    def forward(self, x: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor):
        combined = torch.cat([x, hx], dim=1)
        # Classical gate outputs
        f_lin = torch.sigmoid(self.forget_linear(combined))
        i_lin = torch.sigmoid(self.input_linear(combined))
        g_lin = torch.tanh(self.update_linear(combined))
        o_lin = torch.sigmoid(self.output_linear(combined))

        if self.n_qubits > 0:
            # Quantum gate outputs
            f_q = torch.sigmoid(self.forget_qc(self.forget_q_proj(combined)))
            i_q = torch.sigmoid(self.input_qc(self.input_q_proj(combined)))
            g_q = torch.tanh(self.update_qc(self.update_q_proj(combined)))
            o_q = torch.sigmoid(self.output_qc(self.output_q_proj(combined)))
            # Mix
            f = self.mix_factor * f_q + (1 - self.mix_factor) * f_lin
            i = self.mix_factor * i_q + (1 - self.mix_factor) * i_lin
            g = self.mix_factor * g_q + (1 - self.mix_factor) * g_lin
            o = self.mix_factor * o_q + (1 - self.mix_factor) * o_lin
        else:
            f, i, g, o = f_lin, i_lin, g_lin, o_lin

        cx = f * cx + i * g
        hx = o * torch.tanh(cx)
        return hx, cx

class UnifiedQLSTM(nn.Module):
    """Sequence model that uses the hybrid LSTM cell."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 0, depth: int = 1,
                 mix_factor: float = 0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_cell = UnifiedQLSTMCell(embedding_dim, hidden_dim,
                                          n_qubits=n_qubits, depth=depth,
                                          mix_factor=mix_factor)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        embeds = self.word_embeddings(sentence)
        batch_size = embeds.size(1)
        hx = torch.zeros(batch_size, self.hidden_dim, device=embeds.device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=embeds.device)
        outputs = []
        for x in embeds.unbind(dim=0):
            hx, cx = self.lstm_cell(x, hx, cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        tag_logits = self.hidden2tag(outputs.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["UnifiedQLSTM"]
