from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane import numpy as np
from typing import Tuple, Optional

class QuantumGateBlock(nn.Module):
    """
    A lightweight quantum block that maps a classical vector to a
    probability amplitude vector using a parameterized circuit.
    """
    def __init__(self, input_dim: int, n_qubits: int, depth: int = 2) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.depth = depth
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Parameterized rotation angles
        self.params = nn.Parameter(torch.randn(depth, n_qubits, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten batch dimension for pennylane
        batch_size = x.shape[0]
        out = []

        for i in range(batch_size):
            def circuit(*rotations):
                qml.templates.AngleEmbedding(x[i], wires=range(self.n_qubits))
                for d in range(self.depth):
                    qml.templates.StronglyEntanglingLayers(
                        self.params[d], wires=range(self.n_qubits)
                    )
                return qml.expval(qml.PauliZ(0))

            # Execute circuit and collect expectation
            out.append(qml.execute([self.params], circuit, self.dev, None)[0])

        return torch.stack(out)

class HybridQLSTM(nn.Module):
    """
    Quantum‑classical hybrid LSTM cell built on Pennylane.
    The circuit parameters are trainable and can be jointly optimized
    with the classical gates.  The model supports a tunable
    ``quantum_strength`` coefficient for ablation experiments.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        *,
        quantum_strength: float = 0.0,
        pretrain_quantum: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.quantum_strength = nn.Parameter(
            torch.tensor([quantum_strength], dtype=torch.float32)
        ) if pretrain_quantum else torch.tensor([quantum_strength], dtype=torch.float32)

        # Classical linear gates
        self._lin_forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self._lin_input = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self._lin_update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self._lin_output = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Quantum gate blocks
        self._q_forget = QuantumGateBlock(input_dim + hidden_dim, n_qubits)
        self._q_input = QuantumGateBlock(input_dim + hidden_dim, n_qubits)
        self._q_update = QuantumGateBlock(input_dim + hidden_dim, n_qubits)
        self._q_output = QuantumGateBlock(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            # Classical gate outputs
            f_c = torch.sigmoid(self._lin_forget(combined))
            i_c = torch.sigmoid(self._lin_input(combined))
            g_c = torch.tanh(self._lin_update(combined))
            o_c = torch.sigmoid(self._lin_output(combined))

            # Quantum gate outputs (expectation values in [−1,1])
            f_q = torch.sigmoid(self._q_forget(combined))
            i_q = torch.sigmoid(self._q_input(combined))
            g_q = torch.tanh(self._q_update(combined))
            o_q = torch.sigmoid(self._q_output(combined))

            # Blend gates
            f = (1 - self.quantum_strength) * f_c + self.quantum_strength * f_q
            i = (1 - self.quantum_strength) * i_c + self.quantum_strength * i_q
            g = (1 - self.quantum_strength) * g_c + self.quantum_strength * g_q
            o = (1 - self.quantum_strength) * o_c + self.quantum_strength * o_q

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class HybridLSTMTagger(nn.Module):
    """
    Sequence tagging model that uses the hybrid quantum‑classical LSTM.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        *,
        quantum_strength: float = 0.0,
        pretrain_quantum: bool = False,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = HybridQLSTM(
                embedding_dim,
                hidden_dim,
                n_qubits,
                quantum_strength=quantum_strength,
                pretrain_quantum=pretrain_quantum,
            )
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["HybridQLSTM", "HybridLSTMTagger"]
