import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np

class QuantumSelfAttention(tq.QuantumModule):
    """
    Quantum self‑attention block implemented with a small parameterized
    circuit. The input is encoded into qubit rotations, then processed
    by trainable RX gates and a CNOT chain before measurement.
    """
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_qubits)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_qubits) – the linear projection of the concatenated
        input‑hidden vector onto the qubit space.
        returns: (batch, n_qubits) – measurement outcomes as a classical vector.
        """
        qdev = tq.QuantumDevice(
            n_wires=self.n_qubits,
            bsz=x.shape[0],
            device=x.device,
        )
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        for wire in range(self.n_qubits - 1):
            tqf.cnot(qdev, wires=[wire, wire + 1])
        return self.measure(qdev)

class QLSTM(nn.Module):
    """
    Quantum‑enhanced LSTM cell where each gate is conditioned on a
    quantum self‑attention context derived from the concatenated
    input‑hidden vector.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Map the classical concatenated vector into the qubit space
        self.to_qubits = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.attention = QuantumSelfAttention(n_qubits)

        # Gate linear layers mapping the quantum context to the hidden space
        self.forget_linear = nn.Linear(n_qubits, hidden_dim)
        self.input_linear = nn.Linear(n_qubits, hidden_dim)
        self.update_linear = nn.Linear(n_qubits, hidden_dim)
        self.output_linear = nn.Linear(n_qubits, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            q_input = self.to_qubits(combined)
            attn_out = self.attention(q_input)
            f = torch.sigmoid(self.forget_linear(attn_out))
            i = torch.sigmoid(self.input_linear(attn_out))
            g = torch.tanh(self.update_linear(attn_out))
            o = torch.sigmoid(self.output_linear(attn_out))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
    Sequence tagging model that can switch between the quantum QLSTM
    and a vanilla nn.LSTM by setting the ``n_qubits`` flag.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
