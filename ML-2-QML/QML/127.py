import pennylane as qml
import torch
import torch.nn as nn
from typing import Tuple, Optional


class QLSTMPlus(nn.Module):
    """
    Hybrid LSTM where each gate is computed by a classical linear layer
    followed by a Pennylane quantum circuit that learns a nonlinear feature map.
    """

    class QLayer(nn.Module):
        """
        A lightweight variational circuit implemented with Pennylane.
        It applies a trainable rotation on each wire, entangles the qubits
        with a chain of CNOTs, and measures all qubits in the Z-basis.
        """
        def __init__(self, n_qubits: int, device: str = "default.qubit"):
            super().__init__()
            self.n_qubits = n_qubits
            self.wires = list(range(n_qubits))
            self.theta = nn.Parameter(torch.randn(n_qubits))
            self.qnode = qml.QNode(self.circuit,
                                   qml.device(device, wires=self.wires),
                                   interface="torch")

        def circuit(self, x: torch.Tensor, theta: torch.Tensor):
            # Apply input rotations
            for i in range(self.n_qubits):
                qml.RX(x[i], wires=self.wires[i])
            # Apply trainable rotations
            for i in range(self.n_qubits):
                qml.RX(theta[i], wires=self.wires[i])
            # Entangling CNOT chain
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[self.wires[i], self.wires[i + 1]])
            return [qml.expval(qml.PauliZ(w)) for w in self.wires]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            ``x`` is a tensor of shape (batch, n_qubits) containing
            the classical features to feed into the circuit.
            Returns a tensor of shape (batch, n_qubits) that can be
            back‑propagated through the quantum parameters.
            """
            batch_outputs = []
            for sample in x:
                out = self.qnode(sample, self.theta)
                batch_outputs.append(out)
            return torch.stack(batch_outputs, dim=0)

    def __init__(self, input_dim: int, hidden_dim: int,
                 n_qubits: int = 4, depth: int = 1):
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
                states: Tuple[torch.Tensor, torch.Tensor] | None = None):
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
    Sequence tagging model that uses the hybrid QLSTMPlus with Pennylane.
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
        self.lstm = QLSTMPlus(embedding_dim, hidden_dim,
                              n_qubits=n_qubits, depth=depth)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return torch.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTMPlus", "LSTMTagger"]
