import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple, Optional

class QuantumGateLayer(tq.QuantumModule):
    """
    Shared variational quantum circuit applied to each LSTM gate.
    """
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.params = nn.ParameterList(
            [nn.Parameter(torch.rand(1) * 2 * 3.1415926535) for _ in range(n_qubits)]
        )
        self.cnot_pairs = [(i, (i + 1) % n_qubits) for i in range(n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for i, param in enumerate(self.params):
            tq.RX(param, wires=[i])(qdev)
        for control, target in self.cnot_pairs:
            tqf.cnot(qdev, wires=[control, target])
        return tq.MeasureAll(tq.PauliZ)(qdev)

class QLSTM(nn.Module):
    """
    Hybrid quantum‑classical LSTM cell.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.projection = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.quantum_gate = QuantumGateLayer(n_qubits)
        self.forget_gate = nn.Linear(n_qubits, hidden_dim)
        self.input_gate = nn.Linear(n_qubits, hidden_dim)
        self.update_gate = nn.Linear(n_qubits, hidden_dim)
        self.output_gate = nn.Linear(n_qubits, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_activations: bool = False
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[dict]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        activations = [] if return_activations else None
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            proj = self.projection(combined)
            q_out = self.quantum_gate(proj)
            f = torch.sigmoid(self.forget_gate(q_out))
            i = torch.sigmoid(self.input_gate(q_out))
            g = torch.tanh(self.update_gate(q_out))
            o = torch.sigmoid(self.output_gate(q_out))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
            if return_activations:
                activations.append({"f": f, "i": i, "g": g, "o": o})
        stacked = torch.cat(outputs, dim=0)
        if return_activations:
            return stacked, (hx, cx), activations
        return stacked, (hx, cx), None

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]]
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
    Sequence tagging model that uses the hybrid QLSTM.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int,
        pretrained_embeddings: Optional[torch.Tensor] = None
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            self.word_embeddings.weight.data.copy_(pretrained_embeddings)
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(
        self,
        sentence: torch.Tensor,
        return_activations: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        embeds = self.word_embeddings(sentence)
        lstm_out, _, activations = self.lstm(
            embeds.view(len(sentence), 1, -1),
            return_activations=return_activations
        )
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        log_probs = F.log_softmax(tag_logits, dim=1)
        if return_activations:
            return log_probs, activations
        return log_probs, None

__all__ = ["QLSTM", "LSTMTagger"]
