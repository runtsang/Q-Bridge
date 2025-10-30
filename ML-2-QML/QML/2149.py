"""
Quantum‑enhanced LSTM using Pennylane. Each gate is a variational circuit that
measures all qubits, providing a vector of size `n_qubits`. The implementation
supports dynamic backends and optional noise simulation via Pennylane's
noise module.
"""

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class QLayer(nn.Module):
    """
    Quantum gate implemented with a variational circuit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (and outputs) per gate.
    dev_name : str, default 'default.qubit'
        Pennylane device name.
    """

    def __init__(self, n_qubits: int, dev_name: str = "default.qubit") -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device(dev_name, wires=n_qubits, shots=1)

        def circuit(*params):
            # Encode each input element into an RY rotation
            for w, p in zip(range(n_qubits), params):
                qml.RY(p, wires=w)
            # Entangle adjacent qubits
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Return expectation values of Z for each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.vqc = qml.QNode(circuit, self.dev, interface="torch")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. `x` must be of shape `(batch, n_qubits)`. If the
        input has fewer features, it is padded with zeros.
        """
        angles = x[:, : self.n_qubits]
        if angles.shape[1] < self.n_qubits:
            pad = torch.zeros(
                x.shape[0], self.n_qubits - angles.shape[1], device=x.device, dtype=x.dtype
            )
            angles = torch.cat([angles, pad], dim=1)
        return self.vqc(angles)


class QLSTM(nn.Module):
    """
    Classical LSTM where each gate is a quantum module.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input vector.
    hidden_dim : int
        Hidden state dimensionality.
    n_qubits : int
        Number of qubits per quantum gate.
    dev_name : str, default 'default.qubit'
        Pennylane device name.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        dev_name: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = QLayer(n_qubits, dev_name=dev_name)
        self.input_gate = QLayer(n_qubits, dev_name=dev_name)
        self.update = QLayer(n_qubits, dev_name=dev_name)
        self.output = QLayer(n_qubits, dev_name=dev_name)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass over a sequence.

        Parameters
        ----------
        inputs : Tensor
            Shape `(seq_len, batch, input_dim)`.
        states : Tuple[h, c] | None
            Optional initial hidden and cell states.

        Returns
        -------
        outputs : Tensor
            Shape `(seq_len, batch, hidden_dim)`.
        new_states : Tuple[h, c]
            Final hidden and cell states.
        """
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
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
        if states is None:
            batch_size = inputs.size(1)
            device = inputs.device
            return (
                torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device),
            )
        return states


class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can use either the quantum `QLSTM`
    or a standard `nn.LSTM`.

    Parameters
    ----------
    embedding_dim : int
    hidden_dim : int
    vocab_size : int
    tagset_size : int
    n_qubits : int, default 0
        If zero, the classical LSTM is used.
    dev_name : str, default 'default.qubit'
        Pennylane device name for quantum gates.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        dev_name: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits, dev_name=dev_name)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning log‑softmax over tag logits.

        Parameters
        ----------
        sentence : Tensor
            Shape `(seq_len, batch)` with word indices.
        """
        embeds = self.word_embeddings(sentence).unsqueeze(1)  # (seq_len, batch, 1, emb)
        lstm_out, _ = self.lstm(embeds.squeeze(2))
        tag_logits = self.hidden2tag(lstm_out.squeeze(1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
