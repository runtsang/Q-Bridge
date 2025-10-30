"""Quantum‑enhanced LSTM with a QCNN regression head and fidelity‑based regularisation.

The QML counterpart implements the same public API as the classical module but
uses quantum gates for the LSTM gates and a QCNN‑style variational circuit for
an auxiliary regression loss.  The QCNN is built with Qiskit and executed on a
state‑vector simulator.  The module is intentionally lightweight to keep the
classical‑quantum interface clean.
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import networkx as nx
import itertools
import numpy as np

# --------------------------------------------------------------------------- #
#  Quantum gate layer used for LSTM gates
# --------------------------------------------------------------------------- #

class QuantumGateLayer(tq.QuantumModule):
    """Small variational circuit that acts as a gate on a quantum state."""
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Encode the classical input into a rotation on each qubit
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


# --------------------------------------------------------------------------- #
#  Quantum LSTM
# --------------------------------------------------------------------------- #

class QuantumQLSTM(nn.Module):
    """LSTM where each gate is a quantum circuit."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Quantum gate layers
        self.forget = QuantumGateLayer(n_qubits)
        self.input = QuantumGateLayer(n_qubits)
        self.update = QuantumGateLayer(n_qubits)
        self.output = QuantumGateLayer(n_qubits)

        # Classical linear projections into the qubit space
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=1):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(1))
        stacked = torch.cat(outputs, dim=1)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(0)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


# --------------------------------------------------------------------------- #
#  QCNN‑style regression head
# --------------------------------------------------------------------------- #

class QCNNEvaluator(tq.QuantumModule):
    """A very small QCNN that produces a scalar regression output."""
    def __init__(self, n_qubits: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        # Simple feature map: product of X rotations
        self.feature_map = tq.GeneralEncoder(
            [{"input_idx": [0], "func": "rx", "wires": [0]}]
            + [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(1, n_qubits)]
        )
        self.circuit = tq.QuantumDevice(n_wires=n_qubits)
        # Variational layer
        self.var_layer = tq.RandomLayer(n_ops=20, wires=list(range(n_qubits)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(n_qubits, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=bsz, device=state_batch.device)
        self.feature_map(qdev, state_batch)
        self.var_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


# --------------------------------------------------------------------------- #
#  Hybrid tagger with quantum gate and QCNN head
# --------------------------------------------------------------------------- #

class HybridTagger(nn.Module):
    """Sequence tagger that can switch between classical and quantum LSTM gates
    and optionally uses a QCNN regression head for regularisation.

    The public API mirrors the classical `HybridTagger`.  When `quantum_gate`
    is ``None`` the model behaves classically; otherwise it expects a callable
    that transforms the LSTM output.
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
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QuantumQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        # Optional QCNN head for auxiliary regression loss
        self.qcnn_head = QCNNEvaluator(n_qubits) if n_qubits > 0 else None

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.emb(sentence)
        lstm_out, _ = self.lstm(embeds)
        logits = self.hidden2tag(lstm_out)
        return F.log_softmax(logits, dim=-1)

    # Example of how to use the QCNN head for a regression loss
    def regression_loss(self, hidden_states: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.qcnn_head is None:
            raise ValueError("QCNN head not configured (n_qubits=0).")
        preds = self.qcnn_head(hidden_states.detach())
        return nn.functional.mse_loss(preds, targets)


__all__ = ["QuantumGateLayer", "QuantumQLSTM", "QCNNEvaluator", "HybridTagger"]
