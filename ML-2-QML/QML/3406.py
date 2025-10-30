"""Hybrid quantum LSTM with fully‑connected quantum layer.

- Gates are realized by small quantum circuits implemented with torchquantum.
- The fully connected projection uses a parameterized quantum circuit built on Qiskit.
- The class can operate in a pure quantum mode or fallback to classical nn.Linear if n_qubits == 0.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
import numpy as np

class QLayer(tq.QuantumModule):
    """Quantum gate layer that encodes the input into rotations and applies
    a chain of CNOTs before measuring all qubits.  The output has the same
    dimensionality as the number of wires.
    """
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
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
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        for wire in range(self.n_wires - 1):
            tqf.cnot(qdev, wires=[wire, wire + 1])
        tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
        return self.measure(qdev)

class QuantumFC(nn.Module):
    """Parameterised quantum fully connected layer using a single‑qubit circuit
    per output neuron.  The circuit is a H‑gate followed by a Ry‑rotation
    whose angle is the first input feature of the batch.
    """
    def __init__(self, in_features: int, out_features: int, shots: int = 1024) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

        # Build one circuit per output neuron
        self.circuits: list[tuple[qiskit.QuantumCircuit, qiskit.circuit.Parameter]] = []
        for _ in range(out_features):
            qc = qiskit.QuantumCircuit(1)
            theta = qiskit.circuit.Parameter("theta")
            qc.h(0)
            qc.ry(theta, 0)
            qc.measure_all()
            self.circuits.append((qc, theta))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (batch, in_features)
        batch_size = x.size(0)
        out = torch.zeros(batch_size, self.out_features, device=x.device)
        for i, (qc, theta_param) in enumerate(self.circuits):
            for b in range(batch_size):
                theta_val = x[b, 0].item()
                bound_qc = qc.bind_parameters({theta_param: theta_val})
                job = qiskit.execute(bound_qc, self.backend, shots=self.shots)
                result = job.result()
                counts = result.get_counts(bound_qc)
                probs = np.array(list(counts.values())) / self.shots
                expectation = np.sum(np.array(list(counts.keys()), dtype=float) * probs)
                out[b, i] = expectation
        return out

class HybridQLSTM(nn.Module):
    """Hybrid LSTM cell that can use quantum gates, a quantum fully connected
    output projection, or fall back to classical layers.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        use_quantum_fc: bool = False
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        use_quantum = n_qubits > 0

        # Project the concatenated input+hidden into the dimension used by the
        # quantum gates; this keeps the gate size independent of the raw input.
        self.input_proj = nn.Linear(input_dim + hidden_dim, hidden_dim)

        gate_cls = QLayer if use_quantum else nn.Linear
        self.forget_gate = gate_cls(hidden_dim) if use_quantum else nn.Linear(hidden_dim, hidden_dim)
        self.input_gate = gate_cls(hidden_dim) if use_quantum else nn.Linear(hidden_dim, hidden_dim)
        self.update_gate = gate_cls(hidden_dim) if use_quantum else nn.Linear(hidden_dim, hidden_dim)
        self.output_gate = gate_cls(hidden_dim) if use_quantum else nn.Linear(hidden_dim, hidden_dim)

        proj_cls = QuantumFC if use_quantum_fc else nn.Linear
        self.output_proj = proj_cls(hidden_dim, hidden_dim) if use_quantum_fc else nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            projected = self.input_proj(combined)
            f = torch.sigmoid(self.forget_gate(projected))
            i = torch.sigmoid(self.input_gate(projected))
            g = torch.tanh(self.update_gate(projected))
            o = torch.sigmoid(self.output_gate(projected))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch_size, self.hidden_dim, device=device), \
               torch.zeros(batch_size, self.hidden_dim, device=device)

class LSTMTagger(nn.Module):
    """Tagging model that uses the hybrid quantum LSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_quantum_fc: bool = False
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridQLSTM(
            embedding_dim,
            hidden_dim,
            n_qubits=n_qubits,
            use_quantum_fc=use_quantum_fc
        )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["HybridQLSTM", "LSTMTagger"]
