"""
HybridQLSTM module: classical/quantum LSTM with optional quantum estimator.
Supports sequence tagging and regression tasks.
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the quantum estimator from the companion QML file
# (This import must match the relative location of qml_code)
from.QuantumEstimator import QuantumEstimator


class _QuantumGateLayer(nn.Module):
    """
    Small helper that replaces a linear gate with a quantum module.
    The layer accepts a tensor of shape (batch, dim) and returns a tensor
    of shape (batch, n_qubits) after processing through a quantum circuit.
    """
    def __init__(self, dim: int, n_qubits: int):
        super().__init__()
        self.dim = dim
        self.n_qubits = n_qubits
        # Linear projection to match the number of qubits
        self.linear = nn.Linear(dim, n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project to qubit space
        return self.linear(x)


class HybridQLSTM(nn.Module):
    """
    Hybrid LSTM cell that can use either classical linear gates or quantum gates.
    The gates are realized by a small quantum circuit defined in QuantumEstimator.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        use_quantum_lstm: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_quantum_lstm = use_quantum_lstm

        # Gate definitions
        if self.use_quantum_lstm and self.n_qubits > 0:
            self.forget_gate = _QuantumGateLayer(input_dim + hidden_dim, n_qubits)
            self.input_gate = _QuantumGateLayer(input_dim + hidden_dim, n_qubits)
            self.update_gate = _QuantumGateLayer(input_dim + hidden_dim, n_qubits)
            self.output_gate = _QuantumGateLayer(input_dim + hidden_dim, n_qubits)

            # Quantum modules that act on the projected qubits
            self.forget_qgate = QuantumEstimator(n_qubits)
            self.input_qgate = QuantumEstimator(n_qubits)
            self.update_qgate = QuantumEstimator(n_qubits)
            self.output_qgate = QuantumEstimator(n_qubits)
        else:
            # Classical linear gates
            gate_dim = hidden_dim
            self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
            self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
            self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
            self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Shape (seq_len, batch, input_dim)
        states : Optional[Tuple[torch.Tensor, torch.Tensor]]
            (hx, cx) of shape (batch, hidden_dim)
        """
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            if self.use_quantum_lstm and self.n_qubits > 0:
                # Quantum gate forward
                f = torch.sigmoid(
                    self.forget_qgate(self.forget_gate(combined))
                )
                i = torch.sigmoid(
                    self.input_qgate(self.input_gate(combined))
                )
                g = torch.tanh(
                    self.update_qgate(self.update_gate(combined))
                )
                o = torch.sigmoid(
                    self.output_qgate(self.output_gate(combined))
                )
            else:
                # Classical gates
                f = torch.sigmoid(self.forget_linear(combined))
                i = torch.sigmoid(self.input_linear(combined))
                g = torch.tanh(self.update_linear(combined))
                o = torch.sigmoid(self.output_linear(combined))

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
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


class HybridLSTMTagger(nn.Module):
    """
    Sequence tagging model that can use the hybrid LSTM above.
    Supports both classical and quantum output layers.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_quantum_lstm: bool = True,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridQLSTM(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_qubits=n_qubits,
            use_quantum_lstm=use_quantum_lstm,
        )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        sentence : torch.Tensor
            Shape (seq_len, batch)
        Returns
        -------
        torch.Tensor
            Log-softmax tag logits of shape (seq_len, batch, tagset_size)
        """
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=-1)


class HybridEstimatorQNN(nn.Module):
    """
    Hybrid feed-forward regressor that merges a classical fully-connected network
    with a quantum estimator at the end.  It mirrors the EstimatorQNN example
    but adds a quantum output layer for richer expressivity.
    """
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Tuple[int,...] = (8, 4),
        estimator_qubits: int = 2,
        use_quantum_estimator: bool = True,
    ) -> None:
        super().__init__()
        self.use_quantum_estimator = use_quantum_estimator
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.Tanh())
            prev_dim = h
        self.fc = nn.Sequential(*layers)
        self.out_linear = nn.Linear(prev_dim, 1)

        if self.use_quantum_estimator:
            self.quantum_estimator = QuantumEstimator(estimator_qubits)
        else:
            self.quantum_estimator = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Shape (batch, input_dim)
        Returns
        -------
        torch.Tensor
            Regression output of shape (batch, 1) or (batch, estimator_qubits)
        """
        x = self.fc(inputs)
        linear_out = self.out_linear(x)

        if self.use_quantum_estimator:
            # Map linear output to qubit space via a small linear projection
            proj = nn.Linear(1, self.quantum_estimator.n_qubits).to(linear_out.device)
            qubit_input = proj(linear_out)
            return self.quantum_estimator(qubit_input)
        else:
            return linear_out


__all__ = ["HybridQLSTM", "HybridLSTMTagger", "HybridEstimatorQNN"]
