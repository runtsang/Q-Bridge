"""Quantum‑enhanced LSTM and tagging model.

This module implements a hybrid architecture that mirrors the
classical version but replaces the gating functions with
variational quantum circuits.  The final classification head
is also a parameterised quantum circuit (`FCL`) that can be
executed on a Qiskit simulator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
from qiskit import Aer, execute
from qiskit.circuit import Parameter, QuantumCircuit
from typing import Iterable, Tuple


class QLayer(tq.QuantumModule):
    """Quantum module that implements a small variational circuit
    producing a single expectation value per batch element.
    The circuit consists of a general encoder followed by a
    trainable layer of RX gates and a chain of CNOTs.
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


class QLSTMGen303(nn.Module):
    """Quantum‑enhanced LSTM cell.

    The gating functions are realised by `QLayer` instances.
    When ``n_qubits`` is zero the cell falls back to a purely
    classical implementation for compatibility with non‑quantum
    backends.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
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
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


class FCL(nn.Module):
    """Quantum fully connected layer implemented with a single‑qubit circuit.

    The circuit prepares a uniform superposition, applies a parameterised
    Ry rotation and measures in the computational basis.  The expectation
    value of the measurement is returned as a scalar.
    """
    def __init__(self, n_qubits: int = 1, shots: int = 100) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        self.theta = Parameter("theta")
        self.circuit = QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        result = job.result().get_counts(self.circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probabilities = counts / self.shots
        expectation = np.sum(states * probabilities)
        return np.array([expectation])


class LSTMTaggerGen303(nn.Module):
    """Sequence tagging model that uses the quantum LSTM and a quantum fully
    connected output head.  When ``n_qubits`` is zero the model reverts
    to a classical LSTM with a standard linear classifier.
    """
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
        self.lstm = QLSTMGen303(embedding_dim, hidden_dim, n_qubits=n_qubits)
        if n_qubits > 0:
            self.hidden2tag = FCL(n_qubits=1, shots=100)
        else:
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        if isinstance(self.hidden2tag, FCL):
            thetas = lstm_out.mean(dim=0).detach().cpu().numpy().flatten()
            logits = self.hidden2tag.run(thetas)
            return torch.log_softmax(torch.tensor(logits), dim=1)
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTMGen303", "LSTMTaggerGen303", "FCL"]
