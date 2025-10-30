import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from typing import Tuple, Optional

class QLSTMGen024(nn.Module):
    """
    Quantum‑enhanced LSTM cell. Each gate is a small variational circuit
    followed by a classical linear mapping. Parameters are shared across
    gates to reduce resource usage. A lightweight quantum attention
    mechanism is applied to the hidden state before the output layer.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 4, use_quantum: bool = True) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_quantum = use_quantum

        # Classical linear layers for gate post‑processing
        self.forget_linear = nn.Linear(n_qubits, hidden_dim)
        self.input_linear  = nn.Linear(n_qubits, hidden_dim)
        self.update_linear = nn.Linear(n_qubits, hidden_dim)
        self.output_linear = nn.Linear(n_qubits, hidden_dim)

        # Quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Shared variational parameters for all gates
        self.theta = nn.Parameter(torch.randn(n_qubits, requires_grad=True))

        # Quantum attention device (extra wires)
        self.attn_dev = qml.device("default.qubit", wires=n_qubits + 1)

        # Define quantum gate qnode
        def _quantum_gate(inputs):
            # Encode inputs into qubits via Ry rotations
            for i, val in enumerate(inputs.tolist()):
                qml.RY(val, wires=i)
            # Variational layer
            for i in range(self.n_qubits):
                qml.RY(self.theta[i], wires=i)
            # Entanglement
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Measurement of all qubits
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

        self._quantum_gate = qml.qnode(self.dev, interface="torch")(_quantum_gate)

    def _quantum_gate_batch(self, batch_inputs: torch.Tensor) -> torch.Tensor:
        """
        Apply the quantum gate to each sample in the batch.
        """
        return torch.stack([self._quantum_gate(x) for x in batch_inputs])

    def _quantum_attention(self, h: torch.Tensor) -> torch.Tensor:
        """
        Simple quantum attention that entangles the hidden state with a
        single ancilla qubit and returns a scalar weight.
        """
        def attn_circuit(hidden):
            # Encode hidden into first n_qubits
            for i, val in enumerate(hidden.tolist()):
                qml.RY(val, wires=i)
            # Ancilla entanglement
            qml.CNOT(wires=[0, self.n_qubits])
            # Measurement on ancilla
            return qml.expval(qml.PauliZ(wires=self.n_qubits))
        return qml.qnode(self.attn_dev, interface="torch")(attn_circuit)(h)

    def _init_states(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1) if inputs.dim() == 3 else inputs.size(0)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        seq_len = inputs.size(0) if inputs.dim() == 3 else inputs.size(1)
        for t in range(seq_len):
            x = inputs[t] if inputs.dim() == 3 else inputs[:, t, :]
            combined = torch.cat([x, hx], dim=1)
            # Quantum gates
            f_q = self._quantum_gate_batch(combined)
            i_q = self._quantum_gate_batch(combined)
            g_q = self._quantum_gate_batch(combined)
            o_q = self._quantum_gate_batch(combined)
            # Classical post‑processing
            f = torch.sigmoid(self.forget_linear(f_q))
            i = torch.sigmoid(self.input_linear(i_q))
            g = torch.tanh(self.update_linear(g_q))
            o = torch.sigmoid(self.output_linear(o_q))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            # Quantum attention weight
            attn_weight = self._quantum_attention(hx)
            hx = hx * attn_weight
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

class LSTMTaggerGen024(nn.Module):
    """
    Sequence tagging model that switches between classical LSTM and
    the quantum‑enhanced QLSTMGen024.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 4, use_quantum: bool = True) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if use_quantum and n_qubits > 0:
            self.lstm = QLSTMGen024(embedding_dim, hidden_dim, n_qubits=n_qubits, use_quantum=True)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        sentence: (batch, seq_len)
        """
        embeds = self.word_embeddings(sentence)
        if isinstance(self.lstm, nn.LSTM):
            lstm_out, _ = self.lstm(embeds)
        else:
            lstm_out, _ = self.lstm(embeds.transpose(0,1))
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)

__all__ = ["QLSTMGen024", "LSTMTaggerGen024"]
