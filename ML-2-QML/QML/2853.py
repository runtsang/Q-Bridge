"""UnifiedHybridClassifier: quantum‑enhanced image classifier and sequence tagger.

This module implements the same public API as the classical version
but replaces the dense head with a variational quantum circuit and
the LSTM with a quantum LSTM cell based on torchquantum.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import assemble, transpile

import torchquantum as tq
import torchquantum.functional as tqf

# --------------------------------------------------------------------------- #
# 1.  Quantum circuit wrapper
# --------------------------------------------------------------------------- #
class QuantumCircuitWrapper:
    """Parameterized 2‑qubit circuit executed on Aer."""
    def __init__(self, n_qubits: int, backend, shots: int = 100):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(qubits)
        self.circuit.barrier()
        # Apply a parametrised Ry on each qubit
        for q in qubits:
            self.circuit.ry(self.theta, q)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

# --------------------------------------------------------------------------- #
# 2.  Quantum hybrid head (variational + sigmoid)
# --------------------------------------------------------------------------- #
class QuantumHybridHead(nn.Module):
    """Hybrid head that forwards a single scalar into a quantum circuit."""
    def __init__(self, in_features: int, n_qubits: int, backend, shots: int = 100, shift: float = np.pi/2):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.quantum = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch,)
        params = self.linear(x).squeeze(-1).detach().cpu().numpy()
        # Run quantum circuit
        expectation = self.quantum.run(params)
        probs = torch.tensor(expectation, device=x.device, dtype=x.dtype).squeeze()
        return torch.sigmoid(probs + self.shift)

# --------------------------------------------------------------------------- #
# 3.  Quantum‑enhanced CNN classifier
# --------------------------------------------------------------------------- #
class QuantumCNNClassifier(nn.Module):
    """CNN followed by a quantum hybrid head."""
    def __init__(self, n_qubits: int = 2, shots: int = 200):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = QuantumHybridHead(self.fc3.out_features, n_qubits, backend, shots=shots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.hybrid(x).squeeze(-1)
        return torch.stack([probs, 1 - probs], dim=-1)

# --------------------------------------------------------------------------- #
# 4.  Quantum LSTM cell (torchquantum)
# --------------------------------------------------------------------------- #
class QuantumGateLayer(tq.QuantumModule):
    """Small quantum circuit used as a gate in the quantum LSTM."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)
            ]
        )
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        for wire in range(self.n_wires - 1):
            tqf.cnot(qdev, wires=[wire, wire + 1])
        tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
        return self.measure(qdev)

class QuantumLSTMCell(nn.Module):
    """LSTM cell where each gate is a small quantum circuit."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget_gate = QuantumGateLayer(n_qubits)
        self.input_gate = QuantumGateLayer(n_qubits)
        self.update_gate = QuantumGateLayer(n_qubits)
        self.output_gate = QuantumGateLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, x: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x, hx], dim=1)
        f = torch.sigmoid(self.forget_gate(self.linear_forget(combined)))
        i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
        g = torch.tanh(self.update_gate(self.linear_update(combined)))
        o = torch.sigmoid(self.output_gate(self.linear_output(combined)))
        cx_new = f * cx + i * g
        hx_new = o * torch.tanh(cx_new)
        return hx_new, cx_new

class QuantumLSTMTagger(nn.Module):
    """Sequence tagging model that uses the quantum LSTM cell."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, n_qubits: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_cell = QuantumLSTMCell(embedding_dim, hidden_dim, n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        hx = torch.zeros(sentence.size(1), self.hidden_dim, device=embeds.device)
        cx = torch.zeros(sentence.size(1), self.hidden_dim, device=embeds.device)
        outputs = []
        for x in embeds.unbind(dim=0):
            hx, cx = self.lstm_cell(x, hx, cx)
            outputs.append(hx.unsqueeze(0))
        lstm_out = torch.cat(outputs, dim=0)
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

# --------------------------------------------------------------------------- #
# 5.  UnifiedHybridClassifier (quantum variant)
# --------------------------------------------------------------------------- #
class UnifiedHybridClassifier(nn.Module):
    """Unified model that can act as a quantum classifier or a quantum tagger.

    Parameters
    ----------
    task : str
        One of ``'classification'`` or ``'tagging'``.
    kwargs : dict
        Additional keyword arguments are forwarded to the underlying
        sub‑module constructors.
    """
    def __init__(self, task: str = 'classification', **kwargs) -> None:
        super().__init__()
        if task not in {'classification', 'tagging'}:
            raise ValueError("task must be 'classification' or 'tagging'")
        self.task = task
        if task == 'classification':
            self.model = QuantumCNNClassifier(**kwargs)
        else:
            # kwargs: embedding_dim, hidden_dim, vocab_size, tagset_size, n_qubits
            self.model = QuantumLSTMTagger(**kwargs)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.model(*args, **kwargs)

__all__ = ["UnifiedHybridClassifier"]
