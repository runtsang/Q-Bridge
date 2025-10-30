import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QLSTM(nn.Module):
    """
    Quantum‑augmented LSTM where each gate is implemented by a small variational
    circuit.  The circuit depth and number of qubits are configurable, and a
    noisy simulator can be enabled for experimental robustness studies.
    """

    class QGate(tq.QuantumModule):
        def __init__(self, n_wires: int, depth: int):
            super().__init__()
            self.n_wires = n_wires
            self.depth = depth
            # Parameterised rotation angles (one per wire per depth)
            self.params = nn.ParameterList(
                [nn.Parameter(torch.randn(n_wires)) for _ in range(depth)]
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (batch, n_wires)
            bsz = x.shape[0]
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
            # Encode classical data as rotation angles
            for i in range(self.n_wires):
                tqf.rx(qdev, x[:, i], wires=[i])
            # Variational depth
            for d in range(self.depth):
                for i in range(self.n_wires):
                    tqf.ry(qdev, self.params[d][i], wires=[i])
                for i in range(self.n_wires - 1):
                    tqf.cnot(qdev, wires=[i, i + 1])
            return tq.measure_all(qdev, basis='z')

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, depth: int = 2, noisy: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth
        self.noisy = noisy

        # Linear mapping to qubits
        self.lin_q = nn.Linear(input_dim + hidden_dim, n_qubits * 4)
        # Quantum gates for each LSTM gate
        self.forget_gate = self.QGate(n_qubits, depth)
        self.input_gate = self.QGate(n_qubits, depth)
        self.update_gate = self.QGate(n_qubits, depth)
        self.output_gate = self.QGate(n_qubits, depth)

        # Classical linear layers for state mixing
        self.forget_lin = nn.Linear(n_qubits, hidden_dim)
        self.input_lin = nn.Linear(n_qubits, hidden_dim)
        self.update_lin = nn.Linear(n_qubits, hidden_dim)
        self.output_lin = nn.Linear(n_qubits, hidden_dim)

    def forward(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            # Encode to qubits
            q_in = self.lin_q(combined)
            q_in = q_in.view(-1, self.n_qubits * 4)
            # Gate activations
            f = self.forget_gate(q_in)
            i = self.input_gate(q_in)
            g = self.update_gate(q_in)
            o = self.output_gate(q_in)
            # Map qubit probabilities to gate values
            f = torch.sigmoid(self.forget_lin(f))
            i = torch.sigmoid(self.input_lin(i))
            g = torch.tanh(self.update_lin(g))
            o = torch.sigmoid(self.output_lin(o))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None,
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
    Sequence tagging model that can switch between a classical LSTM and the quantum LSTM.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        depth: int = 2,
        noisy: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(
                embedding_dim,
                hidden_dim,
                n_qubits=n_qubits,
                depth=depth,
                noisy=noisy,
            )
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        if sentence.dim() == 2:
            sentence = sentence.transpose(0, 1)
        embeds = self.word_embeddings(sentence)
        if isinstance(self.lstm, nn.LSTM):
            lstm_out, _ = self.lstm(embeds)
        else:
            lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=-1)


__all__ = ["QLSTM", "LSTMTagger"]
