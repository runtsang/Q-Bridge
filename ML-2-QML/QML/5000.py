"""Hybrid quantum‑classical sequence model.  The module uses a
quantum‑enhanced LSTM (from reference 1), a quantum sampler (from
reference 4) and can optionally prepend a classical auto‑encoder
(from reference 3).  The forward pass is identical to the classical
variant but the internal state updates are performed on a quantum
device."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple

# Quantum LSTM cell (reference 1)
class QLSTM(nn.Module):
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
        return torch.cat(outputs, dim=0), (hx, cx)

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


# Quantum sampler (reference 4)
class QuantumSampler(nn.Module):
    """Wrapper around Qiskit’s SamplerQNN."""

    def __init__(self):
        super().__init__()
        from qiskit_machine_learning.neural_networks import SamplerQNN
        from qiskit.primitives import StatevectorSampler as Sampler

        self.sampler = Sampler()
        self.qnn = SamplerQNN(
            circuit=self._build_circuit(),
            input_params=[],
            weight_params=self._weight_params(),
            sampler=self.sampler,
        )

    def _build_circuit(self):
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2)
        qc.ry(0.0, 0)
        qc.ry(0.0, 1)
        qc.cx(0, 1)
        qc.ry(0.0, 0)
        qc.ry(0.0, 1)
        qc.cx(0, 1)
        qc.ry(0.0, 0)
        qc.ry(0.0, 1)
        return qc

    def _weight_params(self):
        from qiskit.circuit import ParameterVector
        return ParameterVector("weight", 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # The Qiskit SamplerQNN expects a numpy array of shape (n_samples, n_params)
        import numpy as np
        with torch.no_grad():
            probs = self.qnn(inputs.cpu().numpy())
        return torch.from_numpy(probs).to(inputs.device)


class HybridQLSTM(nn.Module):
    """Full hybrid model that chains a classical auto‑encoder, a quantum
    LSTM and a quantum sampler."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        input_dim: int,
        n_qubits: int,
    ) -> None:
        super().__init__()
        # Classical auto‑encoder (reference 3)
        self.autoencoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, input_dim),
        )
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.sampler = QuantumSampler()
        self.classifier = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sentence: LongTensor of shape (seq_len,)
        Returns:
            log‑probabilities of shape (seq_len, tagset_size)
        """
        embeds = self.word_embeddings(sentence)
        # Optional classical compression
        compressed = self.autoencoder(embeds)
        lstm_out, _ = self.lstm(compressed.unsqueeze(1))
        logits = self.classifier(lstm_out.squeeze(1))
        probs = self.sampler(logits)
        return torch.log(probs)


__all__ = ["HybridQLSTM", "QLSTM", "QuantumSampler"]
