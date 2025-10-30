"""
HybridEstimator: Quantum implementation that evaluates Qiskit circuits, supports shot‑sampling,
and includes a quantum LSTMTagger based on torchquantum.  It mirrors the classical
HybridEstimator interface but operates on quantum primitives.
"""

from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, List, Sequence, Optional

import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import cnot


# ----------------------------------------------------------------------
# Quantum sampler (parameterized circuit)
# ----------------------------------------------------------------------
class SamplerQNN:
    """Parameterized quantum sampler that produces a 2‑qubit probability distribution."""
    def __init__(self) -> None:
        self.inputs = ParameterVector("input", 2)
        self.weights = ParameterVector("weight", 4)
        qc = QuantumCircuit(2)
        qc.ry(self.inputs[0], 0)
        qc.ry(self.inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(self.weights[0], 0)
        qc.ry(self.weights[1], 1)
        qc.cx(0, 1)
        qc.ry(self.weights[2], 0)
        qc.ry(self.weights[3], 1)
        self.circuit = qc

    def bind(self,
             input_vals: Sequence[float],
             weight_vals: Sequence[float]) -> QuantumCircuit:
        mapping = {p: v for p, v in zip(self.inputs, input_vals)}
        mapping.update({p: v for p, v in zip(self.weights, weight_vals)})
        return self.circuit.bind_parameters(mapping)


# ----------------------------------------------------------------------
# Hybrid quantum estimator
# ----------------------------------------------------------------------
class HybridEstimator:
    """Quantum estimator that evaluates expectation values of observables
    on a parametrized circuit.  When ``shots`` is given, performs
    shot‑based sampling; otherwise returns deterministic expectations."""
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self,
                 observables: Iterable[BaseOperator],
                 parameter_sets: Sequence[Sequence[float]],
                 *,
                 shots: Optional[int] = None) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound = self._bind(values)
            if shots is None:
                state = Statevector.from_instruction(bound)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                state = Statevector.from_instruction(bound)
                probs = state.probabilities_dict()
                outcomes = np.random.choice(list(probs.keys()), size=shots, p=list(probs.values()))
                row = []
                for obs in observables:
                    # Estimate expectation from sampled states
                    exp = np.mean([state.evolve(QuantumCircuit.from_instruction(obs)).data[0][1] for _ in outcomes])
                    row.append(complex(exp))
            results.append(row)
        return results


# ----------------------------------------------------------------------
# Quantum LSTM / Classical LSTM tagger
# ----------------------------------------------------------------------
class QLSTM(nn.Module):
    """LSTM cell where gates are realised by small quantum circuits."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
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
                    cnot(qdev, wires=[wire, 0])
                else:
                    cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
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

    def forward(self,
                inputs: torch.Tensor,
                states: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
                ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
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

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))


class HybridLSTMTagger(nn.Module):
    """Sequence tagging model that can switch between a classical LSTM and a quantum LSTM."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0) -> None:
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
        return nn.functional.log_softmax(tag_logits, dim=1)


__all__ = ["HybridEstimator", "SamplerQNN", "HybridLSTMTagger"]
