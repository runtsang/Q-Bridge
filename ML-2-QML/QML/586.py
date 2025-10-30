from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QLSTMGen(nn.Module):
    """Quantum‑enhanced LSTM that replaces the classical gates with small parameterised
    quantum circuits and optionally fuses the quantum output with a classical GRU gate.
    The design keeps the forward signature identical to the classical version while
    exposing a feature extractor that maps the qubit measurement results to the hidden
    dimensionality.  The module can be trained end‑to‑end on a quantum simulator
    (CPU or GPU) and supports joint loss weighting between the quantum and classical
    components via the ``quantum_weight`` attribute.
    """
    class QuantumGate(tq.QuantumModule):
        """Parameterised quantum circuit used for each LSTM gate."""
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            # Simple 1‑qubit rotations followed by a small entangling layer
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                    {"input_idx": [2], "func": "rx", "wires": [2]},
                    {"input_idx": [3], "func": "rx", "wires": [3]},
                ]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_qubits)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(
                n_wires=self.n_qubits, bsz=x.shape[0], device=x.device
            )
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_qubits):
                if wire == self.n_qubits - 1:
                    tqf.cnot(qdev, wires=[wire, 0])
                else:
                    tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        feature_extraction: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.feature_extraction = feature_extraction

        # Quantum gates for the LSTM cell
        self.forget_gate = self.QuantumGate(n_qubits)
        self.input_gate = self.QuantumGate(n_qubits)
        self.update_gate = self.QuantumGate(n_qubits)
        self.output_gate = self.QuantumGate(n_qubits)

        # Linear layers to map the concatenated input+hidden state to qubit space
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Classical GRU gate to fuse the quantum‑processed hidden state
        self.gru_gate = nn.GRUCell(hidden_dim, hidden_dim)

        # Optional feature extractor mapping qubit outputs back to hidden_dim
        if feature_extraction:
            self.feature_extractor = nn.Linear(n_qubits, hidden_dim)
        else:
            self.feature_extractor = None

        # Weight controlling the contribution of the quantum loss during training
        self.quantum_weight = 0.5

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            f = torch.sigmoid(self.forget_gate(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
            g = torch.tanh(self.update_gate(self.linear_update(combined)))
            o = torch.sigmoid(self.output_gate(self.linear_output(combined)))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)

            # Classical GRU fusion
            hx = self.gru_gate(hx, hx)

            # Optional feature extraction
            if self.feature_extractor is not None:
                hx = self.feature_extractor(hx)

            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def set_quantum_weight(self, weight: float) -> None:
        """Set the weighting coefficient for the quantum loss component."""
        self.quantum_weight = float(weight)


class LSTMTagger(nn.Module):
    """Sequence tagging model that selects either the quantum QLSTMGen or a classical LSTM.
    The interface is identical to the original implementation, enabling easy experimental
    comparison between the two regimes.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        feature_extraction: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTMGen(
                embedding_dim,
                hidden_dim,
                n_qubits,
                feature_extraction=feature_extraction,
            )
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTMGen", "LSTMTagger"]
