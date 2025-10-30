"""Quantum LSTM implementation using Qiskit circuits and EstimatorQNN."""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator


class ClassicalAngleGenerator(nn.Module):
    """Feed‑forward network that maps classical features to rotation angles."""
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Tanh(),
            nn.Linear(output_dim, output_dim),
            nn.Tanh(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QuantumEstimator(nn.Module):
    """Quantum circuit that evaluates expectation values of a Pauli‑Y observable."""
    def __init__(self, n_qubits: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.input_params = [Parameter(f"input_{i}") for i in range(n_qubits)]
        self.weight_params = [Parameter(f"weight_{i}") for i in range(n_qubits)]
        self.circuit = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            self.circuit.h(i)
            self.circuit.ry(self.input_params[i], i)
            self.circuit.rx(self.weight_params[i], i)
        self.observable = SparsePauliOp.from_list([("Y" * n_qubits, 1)])
        self.estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        # angles: (batch, n_qubits)
        inputs = angles.detach().cpu().numpy()
        preds = self.estimator_qnn.predict(inputs)
        return torch.tensor(preds, device=angles.device, dtype=torch.float)


class QLayerQiskit(nn.Module):
    """Quantum gate implemented with a Qiskit EstimatorQNN."""
    def __init__(self, n_qubits: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.angle_generator = ClassicalAngleGenerator(n_qubits, n_qubits)
        self.quantum_estimator = QuantumEstimator(n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = self.angle_generator(x)
        return self.quantum_estimator(angles)


class QLSTM(nn.Module):
    """Hybrid LSTM with quantum gates implemented via Qiskit."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        gate_dim = hidden_dim

        # Classical linear projections
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

        if n_qubits > 0:
            self.forget_q = QLayerQiskit(n_qubits)
            self.input_q = QLayerQiskit(n_qubits)
            self.update_q = QLayerQiskit(n_qubits)
            self.output_q = QLayerQiskit(n_qubits)
            self.quantum_to_hidden = nn.Linear(n_qubits, hidden_dim)
            self.angle_generator = ClassicalAngleGenerator(input_dim + hidden_dim, n_qubits)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            # Classical gate activations
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))

            if self.n_qubits > 0:
                angles = self.angle_generator(combined)
                f = torch.sigmoid(self.quantum_to_hidden(self.forget_q(angles)))
                i = torch.sigmoid(self.quantum_to_hidden(self.input_q(angles)))
                g = torch.tanh(self.quantum_to_hidden(self.update_q(angles)))
                o = torch.sigmoid(self.quantum_to_hidden(self.output_q(angles)))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)


class LSTMTagger(nn.Module):
    """Sequence tagging model that supports classical or quantum LSTM."""
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
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
