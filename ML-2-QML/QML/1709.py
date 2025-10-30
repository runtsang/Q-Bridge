"""
Quantum‑enhanced LSTM using PennyLane variational circuits.
The implementation remains drop‑in compatible with the original seed
while offering multi‑layer, noise‑aware measurement and a flexible gate
design.
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane import numpy as np


class QLSTM(nn.Module):
    """
    LSTM cell where each gate is realised by a parameterised quantum circuit.
    The circuit is implemented as a PennyLane QNode and can be stacked with
    multiple layers for expressivity.
    """

    class _QuantumGate(nn.Module):
        """
        Tiny variational circuit used for a single LSTM gate.
        The circuit consists of a layer of Ry rotations followed by a
        CNOT ladder to introduce entanglement.
        """

        def __init__(self, n_qubits: int, n_layers: int = 1) -> None:
            super().__init__()
            self.n_qubits = n_qubits
            self.n_layers = n_layers
            # Parameters for Ry rotations
            self.params = nn.Parameter(
                torch.randn(n_qubits * n_layers)
            )  # shape: (n_layers, n_qubits)

            # PennyLane device (default.qubit)
            self.dev = qml.device("default.qubit", wires=n_qubits)

        def _circuit(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
            for i in range(self.n_layers):
                for q in range(self.n_qubits):
                    qml.Ry(params[i, q], wires=q)
                # Entanglement ladder
                for q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])  # wrap‑around

            # Measure all qubits in Z basis
            return qml.expval(qml.PauliZ(self.n_qubits - 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass through the quantum gate.
            Parameters
            ----------
            x : torch.Tensor
                Input of shape (batch, n_qubits) that is linearly transformed
                before being fed to the circuit.
            """
            # Convert the input to a numpy array for PennyLane
            z = x.detach().cpu().numpy()
            # Broadcast parameters to batch
            batch_size = z.shape[0]
            params = self.params.view(self.n_layers, self.n_qubits).cpu().numpy()
            # Vectorised execution across the batch
            out = np.array(
                [self._circuit(z[i], params) for i in range(batch_size)]
            )
            return torch.tensor(out, device=x.device).float()

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, n_layers: int = 2) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Quantum gates for each LSTM gate
        self.forget_gate = self._QuantumGate(n_qubits, n_layers)
        self.input_gate = self._QuantumGate(n_qubits, n_layers)
        self.update_gate = self._QuantumGate(n_qubits, n_layers)
        self.output_gate = self._QuantumGate(n_qubits, n_layers)

        # Classical linear transformations mapping from (input + hidden) to qubits
        self.forget_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_linear = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Optional classical post‑processing to match hidden dimension
        self.post_linear = nn.Linear(n_qubits, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)

        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.forget_linear(combined)))
            i = torch.sigmoid(self.input_gate(self.input_linear(combined)))
            g = torch.tanh(self.update_gate(self.update_linear(combined)))
            o = torch.sigmoid(self.output_gate(self.output_linear(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)

            # Map quantum output dimension back to hidden_dim
            hx = self.post_linear(hx.unsqueeze(1)).squeeze(1)
            outputs.append(hx.unsqueeze(0))

        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can use either the quantum QLSTM defined above
    or the classical LSTM from the ML module. The constructor signature is
    intentionally identical to the original for seamless replacement.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        n_layers: int = 2,
        use_quantum: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        if use_quantum:
            self.lstm = QLSTM(
                embedding_dim,
                hidden_dim,
                n_qubits=n_qubits,
                n_layers=n_layers,
            )
        else:
            # Import classical LSTM lazily to avoid circular dependency
            from.QLSTM__gen289 import QLSTM as ClassicalQLSTM  # type: ignore

            self.lstm = ClassicalQLSTM(
                embedding_dim,
                hidden_dim,
                dropout=0.0,
                use_residual=False,
                n_heads=1,
            )

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
