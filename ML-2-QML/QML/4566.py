"""Quantum‑enhanced LSTM with a fully‑connected quantum module and
FastBaseEstimator.  The design fuses the quantum gates from the QLSTM
seed, the quantum fully‑connected layer from Quantum‑NAT, and the
evaluation utilities from FastBaseEstimator.
"""

from __future__ import annotations

from typing import Iterable, Sequence, List

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class QLayer(tq.QuantumModule):
    """
    Parameterised quantum layer that encodes a feature vector into a small
    register and applies a shallow variational circuit.  The encoder maps each
    element of the input vector to a separate wire via an ``rx`` rotation.
    """

    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor):
        self.encoder(qdev, x)
        for gate, wire in zip(self.params, range(self.n_wires)):
            gate(qdev, wires=wire)
        return self.measure(qdev)


class QuantumFullyConnected(tq.QuantumModule):
    """
    Quantum module that emulates the fully‑connected layer of the
    Quantum‑NAT paper.  It applies a random circuit followed by
    trainable single‑qubit rotations and a controlled‑rx, then measures
    all qubits in the Pauli‑Z basis.
    """

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(n_wires)))
        self.rx0 = tq.RX(has_params=True, trainable=True)
        self.ry0 = tq.RY(has_params=True, trainable=True)
        self.rz0 = tq.RZ(has_params=True, trainable=True)
        self.crx0 = tq.CRX(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor):
        self.encoder(qdev, x)
        self.random_layer(qdev)
        self.rx0(qdev, wires=0)
        self.ry0(qdev, wires=1)
        self.rz0(qdev, wires=3)
        self.crx0(qdev, wires=[0, 2])
        tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
        tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
        tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)
        return self.measure(qdev)


class HybridQLSTM(nn.Module):
    """
    Quantum‑enhanced LSTM that replaces each classical gate with a
    QLayer and can optionally append a QuantumFullyConnected module.
    The interface matches the classical HybridQLSTM class.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_fc: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, input_dim)
        self.n_qubits = n_qubits
        self.use_fc = use_fc

        # Classical linear maps to the quantum register
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        self.forget_gate = QLayer(n_qubits)
        self.input_gate = QLayer(n_qubits)
        self.update_gate = QLayer(n_qubits)
        self.output_gate = QLayer(n_qubits)

        # Projection from quantum register back to hidden dimension
        self.forget_proj = nn.Linear(n_qubits, hidden_dim)
        self.input_proj = nn.Linear(n_qubits, hidden_dim)
        self.update_proj = nn.Linear(n_qubits, hidden_dim)
        self.output_proj = nn.Linear(n_qubits, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        if use_fc:
            self.fc_proj = nn.Linear(hidden_dim, n_qubits)
            self.quantum_fc = QuantumFullyConnected(n_wires=n_qubits)

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

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        hx, cx = self._init_states(embeds)
        outputs = []
        for x in embeds.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f_raw = self.linear_forget(combined)
            i_raw = self.linear_input(combined)
            g_raw = self.linear_update(combined)
            o_raw = self.linear_output(combined)

            qdev_f = tq.QuantumDevice(n_wires=self.n_qubits, bsz=f_raw.shape[0], device=f_raw.device)
            f_q = self.forget_gate(qdev_f, f_raw)
            f = torch.sigmoid(self.forget_proj(f_q))

            qdev_i = tq.QuantumDevice(n_wires=self.n_qubits, bsz=i_raw.shape[0], device=i_raw.device)
            i_q = self.input_gate(qdev_i, i_raw)
            i = torch.sigmoid(self.input_proj(i_q))

            qdev_g = tq.QuantumDevice(n_wires=self.n_qubits, bsz=g_raw.shape[0], device=g_raw.device)
            g_q = self.update_gate(qdev_g, g_raw)
            g = torch.tanh(self.update_proj(g_q))

            qdev_o = tq.QuantumDevice(n_wires=self.n_qubits, bsz=o_raw.shape[0], device=o_raw.device)
            o_q = self.output_gate(qdev_o, o_raw)
            o = torch.sigmoid(self.output_proj(o_q))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        lstm_out = torch.cat(outputs, dim=0)
        tag_logits = self.hidden2tag(lstm_out)
        result = torch.log_softmax(tag_logits, dim=1)

        if self.use_fc:
            fc_input = self.fc_proj(lstm_out[-1]).unsqueeze(0)  # shape (1, n_qubits)
            qdev_fc = tq.QuantumDevice(n_wires=self.n_qubits, bsz=1, device=fc_input.device)
            fc_q = self.quantum_fc(qdev_fc, fc_input)
            result = torch.cat([result, fc_q], dim=1)

        return result


class FastBaseEstimator:
    """
    Quantum estimator that evaluates expectation values of a parametrised circuit.
    """

    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


__all__ = ["HybridQLSTM", "FastBaseEstimator"]
