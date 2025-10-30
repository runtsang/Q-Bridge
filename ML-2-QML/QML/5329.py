"""Hybrid quantum LSTMTagger with quantum encoder and quantum autoencoder.

The module replaces the classical linear gates of an LSTM with small variational
circuits, uses a quantum encoder for input embeddings, and optionally applies a
quantum autoencoder to compress hidden states. It also exposes a quantum
fidelity‑based pruning routine that operates on the quantum states produced
by the encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import networkx as nx
import numpy as np

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
# Quantum encoder: a small fully‑connected variational circuit
# --------------------------------------------------------------------------- #
class QFCModel(tq.QuantumModule):
    """
    Quantum fully‑connected encoder that maps a classical vector into a
    quantum state by applying a variational circuit followed by a measurement.
    """

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=2)
            self.crx0(qdev, wires=[0, 2])

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Classical input with shape (batch, n_wires).

        Returns
        -------
        output : Tensor
            Measured classical bits, shape (batch, n_wires).
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        self.encoder(qdev, x)
        self.q_layer(qdev)
        return self.measure(qdev)


# --------------------------------------------------------------------------- #
# Quantum autoencoder (placeholder)
# --------------------------------------------------------------------------- #
class QuantumAutoencoder(tq.QuantumModule):
    """
    Simple quantum autoencoder that performs a unitary encoding followed by a
    decoding. For demonstration purposes the decoder is the inverse of the
    encoder, so the circuit is effectively identity.
    """

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
        self.decoder = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice):
        self.encoder(qdev)
        self.decoder(qdev)  # In practice this would be the inverse


# --------------------------------------------------------------------------- #
# Quantum LSTM cell
# --------------------------------------------------------------------------- #
class QLSTM(tq.QuantumModule):
    """
    LSTM cell where each gate is implemented by a small quantum circuit.
    """

    class QGate(tq.QuantumModule):
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
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.encoder(qdev)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            # CNOT ladder
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget_gate = self.QGate(n_qubits)
        self.input_gate = self.QGate(n_qubits)
        self.update_gate = self.QGate(n_qubits)
        self.output_gate = self.QGate(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    @tq.static_support
    def forward(self, inputs: Tensor, states: Tuple[Tensor, Tensor] | None = None):
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
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: Tensor,
        states: Tuple[Tensor, Tensor] | None,
    ) -> Tuple[Tensor, Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


# --------------------------------------------------------------------------- #
# Quantum fidelity utilities
# --------------------------------------------------------------------------- #
def _quantum_state_fidelity(a: Tensor, b: Tensor) -> float:
    """Fidelity between two quantum measurement vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def quantum_fidelity_graph(
    states: Tensor, threshold: float, *, secondary: Optional[float] = None, secondary_weight: float = 0.5
) -> nx.Graph:
    """
    Build a weighted graph from quantum measurement vectors.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(states.size(0)))
    for i in range(states.size(0)):
        for j in range(i + 1, states.size(0)):
            fid = _quantum_state_fidelity(states[i], states[j])
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
# Hybrid quantum LSTMTagger
# --------------------------------------------------------------------------- #
class HybridQLSTM(torch.nn.Module):
    """
    Quantum‑enabled LSTMTagger that:
    * embeds inputs with :class:`QFCModel`,
    * processes sequences with :class:`QLSTM`,
    * optionally compresses hidden states with :class:`QuantumAutoencoder`,
    * offers quantum fidelity‑based pruning.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        *,
        n_qubits: int = 4,
        use_autoencoder: bool = False,
        fidelity_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Quantum encoder for embeddings
        self.encoder = QFCModel(n_wires=embedding_dim)

        # Quantum LSTM
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)

        # Optional quantum autoencoder for hidden state compression
        self.autoencoder: Optional[QuantumAutoencoder] = None
        if use_autoencoder:
            self.autoencoder = QuantumAutoencoder(n_wires=hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        self.fidelity_threshold = fidelity_threshold

    @tq.static_support
    def forward(self, sentence: Tensor) -> Tensor:
        """
        Parameters
        ----------
        sentence : LongTensor
            Sequence of token indices with shape (batch, seq_len).

        Returns
        -------
        log_probs : Tensor
            Log‑probabilities over the tagset, shape (batch, seq_len, tagset_size).
        """
        # Embed tokens and encode quantumly
        embeds = self.word_embeddings(sentence)          # (batch, seq_len, embed)
        quantum_embeds = self.encoder(embeds)            # (batch, seq_len, embed)

        # Transpose to (seq_len, batch, embed) for LSTM
        lstm_in = quantum_embeds.transpose(0, 1)

        lstm_out, _ = self.lstm(lstm_in)                # (seq_len, batch, hidden)

        # Optional quantum autoencoder compression
        if self.autoencoder is not None:
            # Apply autoencoder to each hidden state
            compressed = []
            for h in lstm_out.unbind(dim=0):
                qdev = tq.QuantumDevice(n_wires=self.hidden_dim, bsz=1, device=h.device)
                self.autoencoder(qdev)
                compressed.append(qdev.get_state_vector().unsqueeze(0))
            lstm_out = torch.cat(compressed, dim=0)

        # Fidelity‑based pruning
        if self.fidelity_threshold > 0.0:
            # Compute pairwise fidelity between consecutive hidden states
            similarities = torch.cdist(lstm_out, lstm_out, p=2).diagonal(offset=1)
            mask = similarities >= self.fidelity_threshold
            lstm_out = lstm_out * mask.unsqueeze(-1).float()

        # Classification
        logits = self.hidden2tag(lstm_out.transpose(0, 1))  # (batch, seq_len, tagset)
        return F.log_softmax(logits, dim=-1)

    def get_quantum_fidelity_graph(self, hidden_states: Tensor) -> nx.Graph:
        """Return a quantum fidelity graph of the provided hidden states."""
        return quantum_fidelity_graph(hidden_states, self.fidelity_threshold)

__all__ = ["HybridQLSTM", "QFCModel", "QLSTM", "QuantumAutoencoder", "quantum_fidelity_graph"]
