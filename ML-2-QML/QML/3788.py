from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.circuit.library import ZFeatureMap

# -------------------------------------------------------------
# QCNN‑style variational ansatz for the gates
# -------------------------------------------------------------
def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution block used in the QCNN ansatz."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(np.pi / 2, 0)
    return qc

def conv_layer(num_qubits: int, param_prefix: str = "θ") -> QuantumCircuit:
    """Builds a convolutional layer that applies conv_circuit on every pair of qubits."""
    qc = QuantumCircuit(num_qubits)
    param_index = 0
    params = ParameterVector(param_prefix, length=(num_qubits // 2) * 3)
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        layer_circ = conv_circuit(params[param_index:param_index + 3])
        qc.append(layer_circ, [q1, q2])
        param_index += 3
    return qc

# -------------------------------------------------------------
# Quantum LSTM
# -------------------------------------------------------------
class QLSTM(nn.Module):
    """Variational LSTM where every gate is produced by a QCNN‑style ansatz."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Feature map for encoding the classical input
        self.feature_map = ZFeatureMap(input_dim + hidden_dim)

        # Ansatz: a 4‑qubit QCNN layer, one qubit per gate
        self.ansatz = conv_layer(4, param_prefix="θ")
        self.ansatz = self.ansatz.decompose()

        # Observables – one Z on each qubit to read out the four gate values
        obs = SparsePauliOp.from_list([
            ("Z" + "I" * 3, 1),
            ("I" + "Z" * 3, 1),
            ("I" * 2 + "Z" * 2, 1),
            ("I" * 3 + "Z", 1),
        ])

        # Build the EstimatorQNN that returns a 4‑dimensional vector of gate signals
        self.estimator = EstimatorQNN(
            circuit=self.ansatz,
            observables=obs,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=AerSimulator(method="statevector"),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:  # type: ignore[override]
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            # Compute all gates via the quantum circuit
            gates = self.estimator(combined)          # shape (batch, 4)
            f = torch.sigmoid(gates[:, 0])
            i = torch.sigmoid(gates[:, 1])
            g = torch.tanh(gates[:, 2])
            o = torch.sigmoid(gates[:, 3])

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class LSTMTagger(nn.Module):
    """Tagger that uses a quantum‑enhanced LSTM or a classical LSTM."""
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
        self.n_qubits = n_qubits

        if self.n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=self.n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.emb(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
