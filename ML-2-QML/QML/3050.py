import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QLSTM(nn.Module):
    """Quantum‑enhanced LSTM cell where each gate is produced by a small variational circuit."""
    class QLayer(tq.QuantumModule):
        """Produces a scalar per batch item from a vector of qubit amplitudes."""
        def __init__(self, n_wires: int, clip: bool = False):
            super().__init__()
            self.n_wires = n_wires
            self.clip = clip
            # Map classical input to qubit rotations
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            # Trainable rotation angles per wire
            self.params = nn.ParameterList(
                [nn.Parameter((torch.rand(1) * 2 * torch.pi - torch.pi)) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, theta in enumerate(self.params):
                tq.RX(theta, wires=wire).apply(qdev)
            # Entangling CNOT ladder
            for wire in range(self.n_wires - 1):
                tq.CNOT(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim, hidden_dim, n_qubits, clip_params: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.clip_params = clip_params

        # Quantum layers for each gate
        self.forget = self.QLayer(n_qubits, clip=clip_params)
        self.input = self.QLayer(n_qubits, clip=clip_params)
        self.update = self.QLayer(n_qubits, clip=clip_params)
        self.output = self.QLayer(n_qubits, clip=clip_params)

        # Classical linear projections into qubit space
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        if clip_params:
            self._clip_all_params()

    def _clip_all_params(self):
        for gate in [self.forget, self.input, self.update, self.output]:
            for param in gate.params:
                param.data.clamp_(-5.0, 5.0)

    def forward(self, inputs: torch.Tensor, states=None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self, inputs, states):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size,
                 n_qubits=0, clip_params=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits,
                              clip_params=clip_params)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
