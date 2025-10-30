"""Quantum‑centric estimator and LSTM tagger.

The module defines:
* :class:`HybridEstimator` – a lightweight evaluator that accepts a
  Qiskit ``QuantumCircuit`` and a list of ``BaseOperator`` observables.
  It uses a state‑vector simulator to compute expectation values and
  exposes a ``simulate_shots`` helper that adds Gaussian noise to emulate
  finite shot statistics.
* :class:`HybridLSTMTagger` – a sequence tagger that uses a quantum‑gate
  LSTM cell defined in the seed QML code.  The public API mirrors the
  original tagger so that downstream code can switch to the quantum
  version simply by setting ``n_qubits > 0``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

ScalarObservable = Union[BaseOperator, Callable[[torch.Tensor], torch.Tensor | float]]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridEstimator:
    """Quantum‑centric estimator that evaluates a Qiskit circuit for a list of
    parameter sets and observables.  The implementation is a lightweight
    wrapper around Qiskit’s ``Statevector`` simulator.  It also provides a
    ``simulate_shots`` method that adds Gaussian noise to mimic shot noise.
    """
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._params = list(circuit.parameters)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        results: List[List[float]] = []
        for params in parameter_sets:
            bound = self._circuit.assign_parameters(
                dict(zip(self._params, params)), inplace=False
            )
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs).real for obs in observables]
            results.append(row)
        return results

    def simulate_shots(
        self,
        results: List[List[float]],
        shots: int,
        seed: int | None = None,
    ) -> List[List[float]]:
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [rng.normal(loc=val, scale=max(1e-6, 1.0 / shots)) for val in row]
            noisy.append(noisy_row)
        return noisy


# Quantum LSTM cell (seed code)
class QLSTM(nn.Module):
    """Quantum‑gate LSTM cell that uses a small quantum circuit for each gate."""
    class QLayer(nn.Module):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            import torchquantum as tq
            import torchquantum.functional as tqf
            self.tq = tq
            self.tqf = tqf
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(n_wires)
                ]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = self.tq.QuantumDevice(
                n_wires=self.n_wires, bsz=x.shape[0], device=x.device
            )
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                if wire == self.n_wires - 1:
                    self.tqf.cnot(qdev, wires=[wire, 0])
                else:
                    self.tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)
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


class HybridLSTMTagger(nn.Module):
    """Sequence tagging model that can switch between a classical nn.LSTM
    and the quantum‑gate QLSTM.  The public API mirrors the original tagger."""
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
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["HybridEstimator", "HybridLSTMTagger"]
