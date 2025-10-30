import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# Quantum layer that implements a small variational circuit
class QLayer(tq.QuantumModule):
    """Quantum gate that processes a feature vector via RX rotations and CNOTs."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "rx", "wires": [0]},
                {"input_idx": [1], "func": "rx", "wires": [1]},
                {"input_idx": [2], "func": "rx", "wires": [2]},
                {"input_idx": [3], "func": "rx", "wires": [3]},
            ]
        )
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        for wire in range(self.n_wires):
            tgt = 0 if wire == self.n_wires - 1 else wire + 1
            tqf.cnot(qdev, wires=[wire, tgt])
        return self.measure(qdev)

# Quantum LSTM cell where each gate is a variational circuit
class QLSTM(nn.Module):
    """LSTM where forget, input, update, and output gates are realized by QLayer."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = QLayer(n_qubits)
        self.input = QLayer(n_qubits)
        self.update = QLayer(n_qubits)
        self.output = QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def _init_states(self, batch_size: int, device: torch.device):
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(self, inputs: torch.Tensor, states: tuple = None):
        """
        inputs: (seq_len, batch, input_dim)
        """
        seq_len, batch, _ = inputs.size()
        if states is None:
            hx, cx = self._init_states(batch, inputs.device)
        else:
            hx, cx = states
        outputs = []
        for t in range(seq_len):
            x = inputs[t]
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)  # (seq_len, batch, hidden_dim)
        return outputs, (hx, cx)

# Quantum sampler that produces a probability distribution over 4 classes
class QuantumSampler(tq.QuantumModule):
    """
    Variational sampler that accepts a 2‑dim feature vector and a 4‑dim weight vector.
    Returns a log probability over 4 outcomes.
    """
    def __init__(self, input_dim: int = 2, weight_dim: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.weight_dim = weight_dim
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
            ]
        )
        self.weights = nn.Parameter(torch.randn(weight_dim))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: (batch, input_dim)
        Returns log probabilities over 4 outcomes.
        """
        batch = inputs.size(0)
        qdev = tq.QuantumDevice(n_wires=2, bsz=batch, device=inputs.device)
        self.encoder(qdev, inputs)
        # Apply parameterized rotations
        for i in range(2):
            tq.RY(self.weights[i], wires=i)(qdev)
        # Entangle
        tqf.cnot(qdev, wires=[0, 1])
        probs = self.measure(qdev).float()
        # Convert amplitudes to probabilities
        probs = probs / probs.sum(dim=1, keepdim=True)
        return F.log_softmax(probs, dim=1)

class HybridSamplerQLSTM(nn.Module):
    """
    Quantum‑enhanced hybrid sampler.
    Combines a variational QLSTM encoder with a quantum sampler head.
    The public interface mirrors the classical version for API compatibility.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, n_qubits: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.sampler = QuantumSampler()

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        sentence: LongTensor of shape (seq_len,) containing token indices.
        Returns log probabilities over 4 outcomes for each position.
        """
        embeds = self.word_embeddings(sentence).unsqueeze(0)  # (1, seq_len, embed_dim)
        lstm_out, _ = self.lstm(embeds.squeeze(0))
        logits = self.sampler(lstm_out)
        return logits
