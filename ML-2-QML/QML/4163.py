from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.utils import algorithm_globals

from typing import Tuple

from.Autoencoder import Autoencoder
from.QuantumClassifierModel import build_classifier_circuit

class QLayer(tq.QuantumModule):
    """
    Quantum layer that implements a small variational circuit with RX gates
    followed by a CNOT ladder and measurement in the Pauli‑Z basis.
    """

    def __init__(self, n_wires: int) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        for wire in range(self.n_wires):
            if wire == self.n_wires - 1:
                tqf.cnot(qdev, wires=[wire, 0])
            else:
                tqf.cnot(qdev, wires=[wire, wire + 1])
        return self.measure(qdev)

class QLSTM(tq.QuantumModule):
    """
    LSTM cell where gates are realised by small quantum circuits.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
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

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None):
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

class HybridQLSTMClassifier(nn.Module):
    """
    Hybrid LSTM-based sequence tagger that compresses embeddings with a classical
    autoencoder, runs a quantum LSTM layer, and classifies with a variational
    quantum circuit.  The model is fully differentiable through PyTorch.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        latent_dim: int = 32,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        classifier_depth: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.autoencoder = Autoencoder(
            input_dim=embedding_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        if n_qubits > 0:
            self.lstm = QLSTM(latent_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(latent_dim, hidden_dim)

        # Build quantum classifier circuit
        circuit, encoding, weights, observables = build_classifier_circuit(
            num_qubits=hidden_dim, depth=classifier_depth
        )

        def identity_interpret(x: torch.Tensor) -> torch.Tensor:
            return x

        self.qnn = SamplerQNN(
            circuit=circuit,
            input_params=encoding,
            weight_params=weights,
            interpret=identity_interpret,
            output_shape=2,
            sampler=Sampler(),
        )
        self.output_layer = nn.Linear(2, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for sequence tagging.
        :param sentence: Tensor of token indices, shape (seq_len, batch)
        :return: Log-probabilities over tags, shape (seq_len, batch, tagset_size)
        """
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, embedding_dim)
        flat_embeds = embeds.view(-1, embeds.size(-1))
        latent = self.autoencoder.encode(flat_embeds)  # (seq_len*batch, latent_dim)
        latent_seq = latent.view(embeds.size(0), embeds.size(1), -1)

        lstm_out, _ = self.lstm(latent_seq)
        flat_lstm = lstm_out.view(-1, self.hidden_dim)

        # Quantum classifier
        qnn_out = self.qnn(flat_lstm)  # (seq_len*batch, 2)
        logits = self.output_layer(qnn_out)  # (seq_len*batch, tagset_size)
        logits = logits.view(embeds.size(0), embeds.size(1), -1)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQLSTMClassifier"]
