"""Quantum version of HybridQLSTM that uses torchquantum for LSTM gates
and qiskit for the classifier circuit.

The module mirrors the classical interface but replaces the
classical LSTM with a quantum LSTM and the classical classifier
with a parametrised quantum circuit.  FastBaseEstimator is used to
evaluate expectation values on a backend.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.circuit import ParameterVector

# Local imports from the reference pairs
from.FastBaseEstimator import FastBaseEstimator
from.Conv import Conv as QuanvCircuitFactory
from.QuantumClassifierModel import build_classifier_circuit


class QLSTM(nn.Module):
    """LSTM cell where gates are realised by small quantum circuits."""
    class QLayer(tq.QuantumModule):
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


class HybridQLSTM(nn.Module):
    """Quantum hybrid LSTM that combines a quantum LSTM backbone
    with a quantum classifier circuit."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        conv_kernel: int = 2,
        classifier_depth: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Quantum convolutional filter
        self.conv = QuanvCircuitFactory(kernel_size=conv_kernel)

        # Quantum LSTM
        self.lstm = QLSTM(input_dim, hidden_dim, n_qubits=n_qubits)

        # Quantum classifier circuit
        self.classifier_circuit, self.enc, self.wts, self.obs = build_classifier_circuit(n_qubits, classifier_depth)

        # Estimator for expectation values
        self.estimator = FastBaseEstimator(self.classifier_circuit)

        # Linear mapper from hidden state to qubit parameters
        self.param_mapper = nn.Linear(hidden_dim, n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that applies the quantum convolutional filter,
        feeds the result to the quantum LSTM, maps hidden states to
        qubit parameters, and evaluates the classifier circuit.
        """
        batch, seq_len, _ = x.shape
        # Flatten and apply quantum conv filter
        x_flat = x.view(batch * seq_len, -1)
        conv_outputs = []
        kernel_size = int(np.sqrt(self.conv.n_qubits))
        for sample in x_flat.cpu().numpy():
            sample_2d = sample.reshape(kernel_size, kernel_size)
            conv_outputs.append(self.conv.run(sample_2d))
        conv_tensor = torch.tensor(conv_outputs, device=x.device).float()
        conv_tensor = conv_tensor.view(batch, seq_len, -1)

        # Quantum LSTM
        lstm_out, _ = self.lstm(conv_tensor)

        # Map hidden state to qubit parameters
        params = self.param_mapper(lstm_out)  # shape (batch, seq_len, n_qubits)

        # For each batch element, evaluate the classifier circuit
        batch_outputs = []
        for param_set in params.unbind(dim=0):
            bound_circ = self.classifier_circuit.assign_parameters(
                dict(zip(self.enc, param_set.tolist())), inplace=False
            )
            state = Statevector.from_instruction(bound_circ)
            exp_vals = [state.expectation_value(obs) for obs in self.obs]
            batch_outputs.append(exp_vals)
        return torch.tensor(batch_outputs, device=x.device)

    def evaluate(
        self,
        observables: Iterable[SparsePauliOp],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Evaluate the quantum classifier circuit using :class:`FastBaseEstimator`.
        """
        estimator = FastBaseEstimator(self.classifier_circuit)
        return estimator.evaluate(observables, parameter_sets)


class HybridTagger(nn.Module):
    """Sequence tagging wrapper that mirrors the original LSTMTagger API."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.model = HybridQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)

        # Linear layer to map hidden state to tag logits
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        logits = self.model(embeds)
        tag_logits = self.hidden2tag(logits)
        return F.log_softmax(tag_logits, dim=-1)


__all__ = ["HybridQLSTM", "HybridTagger"]
