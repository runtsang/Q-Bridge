'Hybrid estimator with quantum fast evaluation and sequence tagging.\n\nThe quantum variant mirrors the classical implementation but replaces the\n deterministic model evaluation with a Qiskit state‑vector simulator and\n replaces the LSTM with a quantum‑enhanced cell built with torchquantum.\n'

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


class QuantumFastBaseEstimator:
    'Evaluate expectation values of a parametrised Qiskit circuit.'
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError('Parameter count mismatch for bound circuit.')
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)
        return results


class QLSTM(nn.Module):
    'LSTM cell where gates are realised by small quantum circuits.'
    class _QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {'input_idx': [0], 'func': 'rx', 'wires': [0]},
                    {'input_idx': [1], 'func': 'rx', 'wires': [1]},
                    {'input_idx': [2], 'func': 'rx', 'wires': [2]},
                    {'input_idx': [3], 'func': 'rx', 'wires': [3]},
                ]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
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

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        self.forget_gate = self._QLayer(n_qubits)
        self.input_gate = self._QLayer(n_qubits)
        self.update_gate = self._QLayer(n_qubits)
        self.output_gate = self._QLayer(n_qubits)

    def _gate(self, gate: '_QLayer', linear: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        return gate(linear(x))

    def forward(
        self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self._gate(self.forget_gate, self.linear_forget, combined))
            i = torch.sigmoid(self._gate(self.input_gate, self.linear_input, combined))
            g = torch.tanh(self._gate(self.update_gate, self.linear_update, combined))
            o = torch.sigmoid(self._gate(self.output_gate, self.linear_output, combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(
        self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None
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
    'Sequence tagging model that can switch between classical and quantum LSTM.'
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


class HybridEstimator:
    'Unified interface for quantum fast evaluation and sequence tagging.'
    def __init__(
        self,
        circuit: QuantumCircuit | None = None,
        lstm: nn.Module | None = None,
        n_qubits: int = 0,
    ) -> None:
        self.circuit = circuit
        self.lstm = lstm
        if self.circuit is not None:
            self._estimator = QuantumFastBaseEstimator(self.circuit)
        else:
            self._estimator = None
        if self.lstm is None and n_qubits > 0:
            self.lstm = LSTMTagger(
                embedding_dim=50, hidden_dim=64, vocab_size=10000, tagset_size=10, n_qubits=n_qubits
            )

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        if self._estimator is None:
            raise RuntimeError('No circuit supplied for evaluation.')
        return self._estimator.evaluate(observables, parameter_sets)

    def tag(self, sentence: torch.Tensor) -> torch.Tensor:
        if self.lstm is None:
            raise RuntimeError('No LSTM supplied for tagging.')
        return self.lstm(sentence)


__all__ = ['QuantumFastBaseEstimator', 'QLSTM', 'LSTMTagger', 'HybridEstimator']
