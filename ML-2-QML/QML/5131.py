"""Quantum hybrid kernel, classifier, and LSTM tagger."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple, List, Any

import numpy as np
import torch
from torch import nn
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import torch.nn.functional as F

class QuantumRBFKernel(tq.QuantumModule):
    """Quantum kernel using TorchQuantum."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_wires)
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.encoder(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

class Kernel:
    """Wrapper for classical or quantum kernel."""
    def __init__(self, use_quantum: bool = False, gamma: float = 1.0, n_wires: int | None = None):
        self.use_quantum = use_quantum
        self.gamma = gamma
        self.n_wires = n_wires or 4
        if use_quantum:
            self._kernel = QuantumRBFKernel(self.n_wires)
        else:
            self._kernel = None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.use_quantum:
            return self._kernel.forward(x, y)
        else:
            diff = x - y
            return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def gram_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self.forward(x, y).item() for y in b] for x in a])

def build_classifier(num_features: int, depth: int, *, use_quantum: bool = True, n_qubits: int | None = None) -> Tuple[Any, Iterable[int], Iterable[int], List[Any]]:
    if use_quantum:
        encoding = ParameterVector("x", num_qubits=num_features)
        weights = ParameterVector("theta", num_qubits=num_features * depth)

        circuit = QuantumCircuit(num_features)
        for param, qubit in zip(encoding, range(num_features)):
            circuit.rx(param, qubit)

        index = 0
        for _ in range(depth):
            for qubit in range(num_features):
                circuit.ry(weights[index], qubit)
                index += 1
            for qubit in range(num_features - 1):
                circuit.cz(qubit, qubit + 1)

        observables = [SparsePauliOp(f"I" * i + "Z" + "I" * (num_features - i - 1)) for i in range(num_features)]
        return circuit, list(encoding), list(weights), observables
    else:
        layers: List[nn.Module] = []
        in_dim = num_features
        encoding = list(range(num_features))
        weight_sizes: List[int] = []
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.append(linear)
            layers.append(nn.ReLU())
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = num_features
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())
        network = nn.Sequential(*layers)
        observables = list(range(2))
        return network, encoding, weight_sizes, observables

class QLSTM(nn.Module):
    """Quantum-enhanced LSTM cell using TorchQuantum."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]} for i in range(self.n_wires)
                ]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(self.n_wires)])
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

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class LSTMTagger(nn.Module):
    """Sequence tagging model with quantum LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits) if n_qubits > 0 else nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

class HybridKernelClassifier:
    """Unified hybrid kernel, classifier, and optional LSTM tagger."""
    def __init__(self, num_features: int, depth: int, gamma: float = 1.0, n_qubits: int = 4):
        self.kernel = Kernel(use_quantum=True, gamma=gamma, n_wires=n_qubits)
        self.classifier, self.encoding, self.weight_sizes, self.observables = build_classifier(num_features, depth, use_quantum=True, n_qubits=n_qubits)
        self.lstm = QLSTM(num_features, depth, n_qubits=n_qubits)

    def gram_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return self.kernel.gram_matrix(a, b)

    def tag(self, sentence: torch.Tensor) -> torch.Tensor:
        return self.lstm(sentence)

__all__ = ["HybridKernelClassifier", "Kernel", "build_classifier", "QLSTM", "LSTMTagger"]
