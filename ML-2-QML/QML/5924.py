"""Hybrid quantum LSTM with classical feed‑forward head inspired by EstimatorQNN.

The quantum gates are realised with small parameterised circuits
implemented via torchquantum.  The final classification layer is a
quantum EstimatorQNN that returns expectation values used as tag logits.
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
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator


class QuantumGate(tq.QuantumModule):
    """Gate realised by a small variational circuit."""
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


class HybridQLSTM(nn.Module):
    """Quantum LSTM where each gate is a variational circuit."""
    def __init__(self, input_dim: int, hidden_dim: int, n_wires: int = 4) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_wires = n_wires

        self.forget_gate = QuantumGate(n_wires)
        self.input_gate = QuantumGate(n_wires)
        self.update_gate = QuantumGate(n_wires)
        self.output_gate = QuantumGate(n_wires)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_wires)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_wires)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_wires)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_wires)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(
                self.forget_gate(self.linear_forget(combined))
            )
            i = torch.sigmoid(
                self.input_gate(self.linear_input(combined))
            )
            g = torch.tanh(
                self.update_gate(self.linear_update(combined))
            )
            o = torch.sigmoid(
                self.output_gate(self.linear_output(combined))
            )

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


class HybridLSTMTagger(nn.Module):
    """Sequence tagging model that switches between classical and quantum LSTM
    and ends with a quantum EstimatorQNN head for tag logits.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_wires: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_wires > 0:
            self.lstm = HybridQLSTM(embedding_dim, hidden_dim, n_wires=n_wires)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # Quantum EstimatorQNN head
        self.estimator_qnn = self._build_estimator_qnn(tagset_size)

    def _build_estimator_qnn(self, tagset_size: int) -> nn.Module:
        # Build a single‑qubit circuit with a parameterised rotation
        params = [Parameter("theta")]
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.rx(params[0], 0)

        observable = [("Z", 1)]

        estimator = StatevectorEstimator()
        return QiskitEstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[],
            weight_params=[params[0]],
            estimator=estimator,
        )

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        # Convert LSTM hidden states to a flat vector per token
        hidden = lstm_out.view(len(sentence), -1)
        # Pass each hidden vector through the quantum EstimatorQNN
        logits = torch.stack(
            [self.estimator_qnn([h]) for h in hidden],
            dim=0,
        )
        return F.log_softmax(logits, dim=1)


__all__ = ["HybridQLSTM", "HybridLSTMTagger"]
