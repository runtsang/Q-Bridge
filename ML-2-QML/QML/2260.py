"""Quantum‑enhanced hybrid model for sequence tagging and regression.

The quantum variant replaces the classical LSTM gates with small
variational circuits defined in ``tq.QuantumModule``.  It keeps the
same public API – ``HybridQLSTMQuantum`` – so that the classical and
quantum models can be swapped without changing downstream code.

The regression head is identical to the classical version but
operates on the features extracted from a quantum state encoder.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

# --------------------------------------------------------------------------- #
#  Data utilities – quantum superposition dataset
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic quantum regression dataset where each sample is a
    superposition of |0…0⟩ and |1…1⟩ with random angles.

    Parameters
    ----------
    num_wires : int
        Number of qubits (wires).
    samples : int
        Number of samples.

    Returns
    -------
    states : np.ndarray, shape (samples, 2**num_wires)
        Complex‑valued state vectors.
    labels : np.ndarray, shape (samples,)
        Regression targets derived from the angles.
    """
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


class RegressionDataset(torch.utils.data.Dataset):
    """
    Torch ``Dataset`` for the quantum regression data.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
#  Quantum LSTM – gate implementation
# --------------------------------------------------------------------------- #
class QGate(tq.QuantumModule):
    """
    A generic variational gate block that will be used for each LSTM gate.
    It consists of a parameterised RX layer followed by a CNOT chain.
    """
    def __init__(self, n_wires: int):
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


# --------------------------------------------------------------------------- #
#  Quantum LSTM cell
# --------------------------------------------------------------------------- #
class QLSTM(tq.QuantumModule):
    """
    Quantum LSTM cell where each gate is a variational circuit.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget_gate = QGate(n_qubits)
        self.input_gate = QGate(n_qubits)
        self.update_gate = QGate(n_qubits)
        self.output_gate = QGate(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
            g = torch.tanh(self.update_gate(self.linear_update(combined)))
            o = torch.sigmoid(self.output_gate(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

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


# --------------------------------------------------------------------------- #
#  Hybrid model – quantum
# --------------------------------------------------------------------------- #
class HybridQLSTMQuantum(tq.QuantumModule):
    """
    Quantum‑enhanced hybrid model that performs sequence tagging and
    regression.  It uses ``QLSTM`` for the tagging part and a quantum
    encoder + feed‑forward head for regression.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        num_wires: int,
        n_qubits: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.tag_head = nn.Linear(hidden_dim, tagset_size)

        # Quantum encoder for regression
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = QGate(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.reg_head = nn.Linear(num_wires, 1)

    def forward(
        self,
        sentence: torch.Tensor,
        state_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        sentence : torch.Tensor, shape (batch, seq_len)
            Token indices for tagging.
        state_batch : torch.Tensor, shape (batch, 2**num_wires)
            Quantum state vectors for regression.

        Returns
        -------
        tag_loss : torch.Tensor
            Log‑softmax tag logits.
        reg_out : torch.Tensor
            Regression predictions.
        """
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.tag_head(lstm_out)
        tag_loss = F.log_softmax(tag_logits, dim=-1)

        # Regression path
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        reg_out = self.reg_head(features).squeeze(-1)
        return tag_loss, reg_out


__all__ = ["HybridQLSTMQuantum", "RegressionDataset", "generate_superposition_data"]
