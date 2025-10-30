"""Hybrid QCNN-QLSTM model – quantum implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
import torchquantum as tq
import torchquantum.functional as tqf

__all__ = ["UnifiedQCNNQLSTM", "UnifiedQCNNQLSTMConfig"]

# Quantum building blocks

def conv_circuit(params):
    """Two‑qubit convolution circuit used in QCNN."""
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

def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Quantum convolutional layer acting on all qubits in pairs."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index:param_index+3]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index:param_index+3]), [q1, q2])
        qc.barrier()
        param_index += 3
    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc

def pool_circuit(params):
    """Two‑qubit pooling circuit used in QCNN."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def pool_layer(sources, sinks, param_prefix: str) -> QuantumCircuit:
    """Quantum pooling layer that maps source qubits to sink qubits."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index:param_index+3]), [source, sink])
        qc.barrier()
        param_index += 3
    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc

class UnifiedQCNNQLSTMConfig:
    """Configuration for the hybrid QCNN-QLSTM quantum model."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 16,
        qcnn_layers: int = 3,
        n_qubits_lstm: int | None = None,
        lstm_layers: int = 1,
        dropout: float = 0.0,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.qcnn_layers = qcnn_layers
        self.n_qubits_lstm = n_qubits_lstm if n_qubits_lstm is not None else hidden_dim
        self.lstm_layers = lstm_layers
        self.dropout = dropout

class UnifiedQCNNQLSTM(nn.Module):
    """Quantum hybrid QCNN‑QLSTM model.

    The model first embeds each input vector with a Z‑feature map,
    then applies a stack of quantum convolution‑pooling blocks
    (QCNN).  The resulting quantum state is measured and used as
    input to a quantum‑enhanced LSTM cell.  The hidden state of the
    LSTM is propagated across the sequence and a linear head produces
    the final output.

    Parameters
    ----------
    config : UnifiedQCNNQLSTMConfig or dict
        Configuration object or dictionary.
    """

    def __init__(self, config: UnifiedQCNNQLSTMConfig | dict | None = None):
        super().__init__()
        if config is None:
            config = UnifiedQCNNQLSTMConfig()
        if isinstance(config, dict):
            config = UnifiedQCNNQLSTMConfig(**config)
        self.config = config

        # Feature map
        self.feature_map = ZFeatureMap(self.config.input_dim)

        # QCNN ansatz
        self.qcnn_ansatz = self._build_qcnn_ansatz()

        # QNN for QCNN
        self.qcnn_qnn = EstimatorQNN(
            circuit=self.qcnn_ansatz.decompose(),
            observables=SparsePauliOp.from_list([("Z" * self.config.input_dim, 1)]),
            input_params=self.feature_map.parameters,
            weight_params=self.qcnn_ansatz.parameters,
            estimator=Estimator(),
        )

        # Linear mapping from QCNN scalar output to hidden dimension
        self.qcnn_to_hidden = nn.Linear(1, self.config.hidden_dim)

        # Quantum LSTM
        self.lstm = QLSTMQuantum(
            input_dim=self.config.hidden_dim,
            hidden_dim=self.config.hidden_dim,
            n_qubits=self.config.n_qubits_lstm,
        )

        # Output head
        self.head = nn.Linear(self.config.hidden_dim, 1)

    def _build_qcnn_ansatz(self) -> QuantumCircuit:
        """Construct the QCNN ansatz by stacking convolution layers."""
        ansatz = QuantumCircuit(self.config.input_dim)
        for i in range(self.config.qcnn_layers):
            ansatz.compose(conv_layer(self.config.input_dim, f"c{i+1}"), inplace=True)
        return ansatz

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input shape (batch, seq_len, input_dim).

        Returns
        -------
        torch.Tensor
            Output shape (batch, 1).
        """
        batch, seq_len, _ = x.shape
        # QCNN features per time step
        seq_features = []
        for t in range(seq_len):
            xt = x[:, t, :]
            # Feature map embedding
            feat = self.feature_map(xt)
            # QCNN QNN
            qnn_out = self.qcnn_qnn(feat)  # shape (batch, 1)
            hidden = self.qcnn_to_hidden(qnn_out)  # shape (batch, hidden_dim)
            seq_features.append(hidden.unsqueeze(0))
        # Shape: (seq_len, batch, hidden_dim)
        seq_features = torch.cat(seq_features, dim=0)
        # LSTM
        lstm_out, _ = self.lstm(seq_features)
        # Take last hidden state
        last_hidden = lstm_out[-1, :, :]
        out = self.head(last_hidden)
        return torch.sigmoid(out)

class QLSTMQuantum(nn.Module):
    """Quantum‑enhanced LSTM cell using small quantum circuits for gates."""

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
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
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
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

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
