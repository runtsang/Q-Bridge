"""Quantum‑enhanced LSTM with a variational forget gate implemented in PennyLane."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import pennylane.numpy as pnp


class HybridQLSTM(nn.Module):
    """A classical LSTM cell that uses a variational quantum circuit for the forget gate.

    Parameters
    ----------
    input_dim : int
        Dimension of input embeddings.
    hidden_dim : int
        Hidden state size.
    n_qubits : int
        Number of qubits for the quantum circuit.
    use_advanced_circuit : bool, default=False
        If True, use a deeper ansatz with entanglement; otherwise a single‑wire RX.
    """

    class _QuantumForgetGate(nn.Module):
        """Variational quantum circuit for the forget gate."""
        def __init__(self, n_qubits: int, use_advanced: bool):
            super().__init__()
            self.n_qubits = n_qubits
            self.use_advanced = use_advanced
            dev = qml.device("default.qubit", wires=n_qubits, shots=1024)
            self.qnode = qml.QNode(self._circuit, dev)

        def _circuit(self, x: pnp.ndarray):
            # Encode classical data as RX rotations
            for i in range(self.n_qubits):
                qml.RX(x[i], wires=i)
            if self.use_advanced:
                # Entangle with CNOT ladder
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                # Additional parameterised rotations
                for i in range(self.n_qubits):
                    qml.RY(0.5, wires=i)
            # Measurement
            return qml.expval(qml.PauliZ(0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, n_qubits)
            # Convert to numpy array for PennyLane
            x_np = x.detach().cpu().numpy()
            batch_out = []
            for sample in x_np:
                val = self.qnode(sample)
                batch_out.append(val)
            out = torch.tensor(batch_out, device=x.device, dtype=torch.float32)
            return out.unsqueeze(1)  # (batch, 1)

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        use_advanced_circuit: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_advanced_circuit = use_advanced_circuit

        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

        self.quantum_forget = self._QuantumForgetGate(n_qubits, use_advanced_circuit)
        self.quantum_proj = nn.Linear(gate_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f_class = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))

            # Project to qubit space and run quantum circuit
            q_input = torch.tanh(self.quantum_proj(combined))
            q_out = self.quantum_forget(q_input)
            # Scale quantum output to [0,1] by sigmoid
            f_q = torch.sigmoid(q_out)
            # Combine classical and quantum forget gates
            f = 0.5 * f_class + 0.5 * f_q

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


class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_advanced_circuit: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = HybridQLSTM(embedding_dim, hidden_dim, n_qubits, use_advanced_circuit)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["HybridQLSTM", "LSTMTagger"]
