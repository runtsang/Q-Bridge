"""
Quantum‑enhanced LSTM tagging model with a variational quantum circuit
for the gates and an optional quantum estimator head for tag logits.
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator


class QLayer(tq.QuantumModule):
    """
    Variational quantum layer that applies parameterised rotations followed by a
    linear‑circuit of CNOTs. The number of wires equals the number of qubits
    used to encode a gate output.
    """

    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires

        # Simple encoding: each input feature controls an RX rotation on its wire
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [idx], "func": "rx", "wires": [idx]}
                for idx in range(n_wires)
            ]
        )

        # Trainable rotation gates on each wire
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )

        # Entangling layer – a chain of CNOTs
        self.entangle = [
            tqf.cnot if i < n_wires - 1 else tqf.cnot
            for i in range(n_wires)
        ]

        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute the circuit for a batch of inputs and return expectation values.
        """
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        for wire in range(self.n_wires - 1):
            tqf.cnot(qdev, wires=[wire, wire + 1])
        tqf.cnot(qdev, wires=[self.n_wires - 1, 0])  # wrap‑around
        return self.measure(qdev)


class QLSTM(nn.Module):
    """
    Quantum‑enhanced LSTM cell where each gate is realised by a small variational
    quantum circuit. The linear projection maps the classical concatenated
    state onto the qubit space before feeding it to the gate circuit.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = QLayer(n_qubits)
        self.input_gate = QLayer(n_qubits)
        self.update = QLayer(n_qubits)
        self.output = QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
        return torch.cat(outputs, dim=0), (hx, cx)

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


class QLSTMTagger(nn.Module):
    """
    Sequence‑tagging model that uses a quantum‑enhanced LSTM and a classical linear head.
    The design is fully compatible with the classical API while exposing quantum gates
    for research into hybrid architectures.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that uses the quantum LSTM and returns log‑softmaxed tag scores.
        """
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


def EstimatorQNN() -> QiskitEstimatorQNN:
    """
    Quantum estimator that mirrors the simple QNN example but is built with
    Qiskit and can be used as a head for the :class:`QLSTMTagger`.
    """
    # Define a single‑qubit circuit with a variable rotation and a measurement
    params = [Parameter("input"), Parameter("weight")]
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.ry(params[0], 0)
    qc.rx(params[1], 0)

    observable = SparsePauliOp.from_list([("Y", 1)])

    estimator = StatevectorEstimator()
    return QiskitEstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=[params[0]],
        weight_params=[params[1]],
        estimator=estimator,
    )


__all__ = ["QLSTM", "QLSTMTagger", "EstimatorQNN"]
