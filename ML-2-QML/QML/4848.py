"""Hybrid quantum model combining quantum classifier, quantum LSTM tagger, and quantum regression."""
from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
from torch.utils.data import Dataset

# ----------------------------------------------------------------------
# 1. Quantum classifier circuit
# ----------------------------------------------------------------------
def build_classifier_circuit(num_qubits: int, depth: int, use_mixed: bool = False) -> Tuple[tq.QuantumCircuit, Iterable[tq.ParameterVector], Iterable[tq.ParameterVector], list[tq.SparsePauliOp]]:
    """
    Layered quantum ansatz with optional mixed encoding.
    Returns the circuit, two lists of ParameterVectors (encoding & weights),
    and a list of Pauli‑Z observables for measurement.
    """
    encoding = tq.ParameterVector("x", num_qubits)
    weights = tq.ParameterVector("theta", num_qubits * depth)
    circuit = tq.QuantumCircuit(num_qubits)

    # Data encoding
    for p, q in zip(encoding, range(num_qubits)):
        circuit.rx(p, q)

    # Ansatz layers
    idx = 0
    for _ in range(depth):
        for q in range(num_qubits):
            circuit.ry(weights[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            circuit.cz(q, q + 1)

    # Optional mixed encoding
    if use_mixed:
        for q in range(num_qubits):
            circuit.rz(tq.ParameterVector(f"phi_{q}", 1)[0], q)

    observables = [tq.SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

# ----------------------------------------------------------------------
# 2. Quantum LSTM layers
# ----------------------------------------------------------------------
class QLSTM(nn.Module):
    """
    Quantum‑enhanced LSTM cell that maps combined classical state to a
    quantum register, applies a variational block, measures, and feeds
    the result back into the classic recurrence.
    """
    class QGate(tq.QuantumModule):
        def __init__(self, n_wires: int):
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
            for w, gate in enumerate(self.params):
                gate(qdev, wires=w)
            for w in range(self.n_wires):
                tqf.cnot(qdev, wires=[w, (w + 1) % self.n_wires])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = self.QGate(n_qubits)
        self.input = self.QGate(n_qubits)
        self.update = self.QGate(n_qubits)
        self.output = self.QGate(n_qubits)

        self.lin_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        self.linear_forget = nn.Linear(n_qubits, hidden_dim)
        self.linear_input = nn.Linear(n_qubits, hidden_dim)
        self.linear_update = nn.Linear(n_qubits, hidden_dim)
        self.linear_output = nn.Linear(n_qubits, hidden_dim)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f_q = self.forget(self.lin_forget(combined))
            i_q = self.input(self.lin_input(combined))
            g_q = self.update(self.lin_update(combined))
            o_q = self.output(self.lin_output(combined))

            f = torch.sigmoid(self.linear_forget(f_q))
            i = torch.sigmoid(self.linear_input(i_q))
            g = torch.tanh(self.linear_update(g_q))
            o = torch.sigmoid(self.linear_output(o_q))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class LSTMTagger(nn.Module):
    """Sequence tagging model that uses a quantum LSTM if ``n_qubits`` > 0."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, n_qubits: int = 0):
        super().__init__()
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
        return torch.log_softmax(tag_logits, dim=1)

# ----------------------------------------------------------------------
# 3. Quantum regression dataset and model
# ----------------------------------------------------------------------
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate superposition states and target labels."""
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)

    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1

    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(tq.QuantumModule):
    """Quantum regression model with a variational layer followed by a classical head."""
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

# ----------------------------------------------------------------------
# 4. Unified wrapper
# ----------------------------------------------------------------------
class HybridModel(tq.QuantumModule):
    """
    Wrapper that selects the appropriate quantum sub‑module based on ``mode``.
    Modes: 'classifier','regression', 'tagger'.
    """
    def __init__(self, mode: str, **kwargs):
        super().__init__()
        self.mode = mode
        if mode == "classifier":
            num_qubits = kwargs["num_qubits"]
            depth = kwargs["depth"]
            self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(num_qubits, depth)
        elif mode == "regression":
            num_wires = kwargs["num_wires"]
            self.model = QModel(num_wires)
        elif mode == "tagger":
            embedding_dim = kwargs["embedding_dim"]
            hidden_dim = kwargs["hidden_dim"]
            vocab_size = kwargs["vocab_size"]
            tagset_size = kwargs["tagset_size"]
            n_qubits = kwargs.get("n_qubits", 0)
            self.model = LSTMTagger(embedding_dim, hidden_dim, vocab_size, tagset_size, n_qubits=n_qubits)
        else:
            raise ValueError(f"Unsupported mode {mode}")

    def forward(self, *args, **kwargs):
        if self.mode == "classifier":
            raise NotImplementedError("Forward for classifier mode is not implemented in the high‑level wrapper.")
        else:
            return self.model(*args, **kwargs)

__all__ = [
    "build_classifier_circuit",
    "QLSTM",
    "LSTMTagger",
    "RegressionDataset",
    "QModel",
    "HybridModel",
    "generate_superposition_data",
]
