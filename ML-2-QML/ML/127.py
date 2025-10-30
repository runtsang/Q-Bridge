import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class QLSTMPlus(nn.Module):
    """
    Hybrid LSTM where each gate is computed by a classical linear layer
    followed by a *parameter‑shaped* quantum circuit that acts on a
    small number of qubits. The quantum module is `QLayer`, a lightweight
    variational circuit that can be turned on/off independently per gate.
    """

    class QLayer(nn.Module):
        """
        A tiny variational circuit that applies a trainable rotation
        on each wire and then a chain of CNOTs, producing a classical
        feature vector of size ``n_qubits``.  The circuit is
        intentionally shallow to allow fast simulation.
        """
        def __init__(self, n_qubits: int, device: Optional[str] = None):
            super().__init__()
            self.n_qubits = n_qubits
            self.device = device or torch.device('cpu')
            # Trainable rotation angles
            self.rxs = nn.Parameter(torch.randn(n_qubits))
            # Classical linear transform before measurement
            self.pre = nn.Linear(n_qubits, n_qubits, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass of the quantum layer.
            ``x`` is expected to shape (batch, n_qubits).
            Returns a torch tensor of the same shape that can be
            back‑propagated through the parameterised rotations.
            """
            # Simulate the rotation via a matrix product
            cos = torch.cos(self.rxs)
            sin = torch.sin(self.rxs)
            rot = torch.stack([torch.stack([cos, -sin], dim=2),
                               torch.stack([sin, cos], dim=2)], dim=1)
            # Apply per‑wire rotation
            # (batch, n_quw, 2, 2)
            x = x.unsqueeze(-1)  # (batch, n_quw, 1)
            matrix = rot.repeat(1, 1, 1, 1)
            result = torch.matmul(x, matrix).squeeze(-1)
            # Chain of CNOTs (simulated by a linear layer)
            result = self.pre(result)
            return result

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 4, depth: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth

        # Classical linear gate network
        self.fc_forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.fc_input  = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.fc_update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.fc_output = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Quantum gate encoders
        self.quantum = nn.ModuleDict({
            'forget': self.QLayer(n_qubits),
            'input':  self.QLayer(n_qubits),
            'update': self.QLayer(n_qubits),
            'output': self.QLayer(n_qubits)
        })

    def forward(self, inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.quantum['forget'](self.fc_forget(combined)))
            i = torch.sigmoid(self.quantum['input'](self.fc_input(combined)))
            g = torch.tanh(self.quantum['update'](self.fc_update(combined)))
            o = torch.sigmoid(self.quantum['output'](self.fc_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
    Sequence tagging model that uses the hybrid QLSTMPlus.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 4,
        depth: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTMPlus(embedding_dim, hidden_dim, n_qubits=n_qubits, depth=depth)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTMPlus", "LSTMTagger"]
