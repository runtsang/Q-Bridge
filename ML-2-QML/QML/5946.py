"""Quantum‑enhanced LSTM tagging model with a variational classifier head.

The module mirrors the classical implementation but replaces the
LSTM and classifier with quantum‑aware counterparts.  The
`QLSTMClassifier` class can be instantiated with `n_qubits > 0` to
activate the quantum LSTM cell, and with
`classifier_n_qubits > 0` to activate a variational quantum
classifier.  Falling back to the classical backbone is also
supported for rapid prototyping.

The implementation uses `torchquantum` for the quantum LSTM and
a custom variational circuit, while the quantum classifier builder
leverages a lightweight `qiskit` ansatz that can be updated
incrementally.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Quantum LSTM core
import torchquantum as tq
import torchquantum.functional as tqf

# Quantum classifier circuit builder
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_classifier_circuit(
    num_qubits: int, depth: int
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a layered ansatz with explicit encoding and variational
    parameters, mirroring the classical feed‑forward network.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the classifier circuit.
    depth : int
        Number of variational layers.

    Returns
    -------
    Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]
        * Quantum circuit object
        * List of encoding parameters
        * List of variational parameters
        * Observables (one Pauli‑Z per qubit)
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    index = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables


class QClassifier(tq.QuantumModule):
    """
    Variational quantum classifier that maps a feature vector to a
    probability distribution over two classes.  The circuit consists of
    an input encoding with `RX` gates followed by `RY` variational
    layers and `CZ` entangling gates.
    """

    def __init__(self, n_wires: int, depth: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        # Trainable parameters for the variational `RY` gates
        self.ry_params = nn.Parameter(torch.randn(depth, n_wires))

        # Measurement of all qubits in the Pauli‑Z basis
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch, n_wires).

        Returns
        -------
        torch.Tensor
            Expectation values of the Pauli‑Z operators (one per qubit).
        """
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=x.shape[0], device=x.device
        )
        # Encoding with RX gates driven by the input features
        for wire in range(self.n_wires):
            tq.RX(x[:, wire].unsqueeze(-1))(qdev, wires=wire)

        # Variational layers
        for d in range(self.depth):
            for wire in range(self.n_wires):
                tq.RY(self.ry_params[d, wire].unsqueeze(0))(qdev, wires=wire)
            for wire in range(self.n_wires - 1):
                tq.CZ(qdev, wires=[wire, wire + 1])

        # Return measurement outcomes
        return self.measure(qdev)


class QLSTM(nn.Module):
    """
    Quantum LSTM cell where each gate is realised by a small
    parameterised quantum circuit.  The circuit is identical for all
    gates but is wrapped in a dedicated `QLayer` class that handles
    encoding, trainable rotations and entanglement.
    """

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            # Encoder that applies a single RX gate per wire
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(n_wires)
                ]
            )
            # Trainable rotation gates
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            # All‑qubit measurement in the Pauli‑Z basis
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(
                n_wires=self.n_wires, bsz=x.shape[0], device=x.device
            )
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                if wire == self.n_wires - 1:
                    tqf.cnot(qdev, wires=[wire, 0])
                else:
                    tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Quantum gates for each LSTM component
        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        # Linear projections into the quantum space
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(0)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


class QLSTMClassifier(nn.Module):
    """
    Hybrid sequence‑tagging model that can operate in classical or
    quantum mode.  The quantum mode uses a `QLSTM` cell and a
    variational `QClassifier` head; the classical mode falls back to
    PyTorch’s `nn.LSTM` and a feed‑forward classifier.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        classifier_depth: int = 2,
        classifier_n_qubits: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # LSTM backbone
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # Classifier head
        if classifier_n_qubits and classifier_n_qubits > 0:
            self.classifier = QClassifier(
                n_wires=classifier_n_qubits, depth=classifier_depth
            )
        else:
            # Use the classical feed‑forward network from the ML module
            self.classifier_network, _, _, _ = build_classifier_circuit(
                num_qubits=hidden_dim, depth=classifier_depth
            )

        # Tag projection
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for sequence tagging.

        Parameters
        ----------
        sentence : torch.Tensor
            Tensor of word indices with shape (batch, seq_len).

        Returns
        -------
        torch.Tensor
            Log‑probabilities over tags for each token.
        """
        embeds = self.word_embeddings(sentence)
        if isinstance(self.lstm, nn.LSTM):
            lstm_out, _ = self.lstm(embeds)
        else:
            lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)

        # Optional quantum/classical classification (not used in tag output)
        if hasattr(self, "classifier"):
            if isinstance(self.classifier, QClassifier):
                _ = self.classifier(lstm_out)
            else:
                _ = self.classifier_network(lstm_out)

        return F.log_softmax(tag_logits, dim=2)

__all__ = ["QLSTMClassifier", "build_classifier_circuit", "QLSTM", "QClassifier"]
