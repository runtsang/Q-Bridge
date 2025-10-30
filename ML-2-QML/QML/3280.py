"""Quantum‑only implementation of the hybrid QCNN‑QLSTM architecture.

The module contains:
  - QuantumQCNN: a Qiskit based variational circuit that reproduces the
    convolution‑pool hierarchy of the classical QCNN seed.
  - QuantumQLSTM: a TorchQuantum LSTM cell where each gate is a small
    parameterised quantum circuit.
  - UnifiedQCNNQLSTM: an end‑to‑end model that chains QuantumQCNN with
    QuantumQLSTM for sequence tagging.  The API mirrors the classical
    counterpart so that the two modules can be swapped interchangeably.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

# Quantum QCNN implementation
class QuantumQCNN(nn.Module):
    """Qiskit based variational circuit that implements convolution and pooling layers."""
    def __init__(self, n_qubits: int = 8) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.feature_map = ZFeatureMap(n_qubits, reps=1)
        self.circuit = self._build_ansatz()

        # Estimator for state‑vector simulation
        self.estimator = StatevectorEstimator()

        # Convert circuit to EstimatorQNN
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=[SparsePauliOp.from_list([("Z" + "I" * (n_qubits - 1), 1)])],
            input_params=self.feature_map.parameters,
            weight_params=self.circuit.parameters,
            estimator=self.estimator,
        )

    def _build_ansatz(self) -> QuantumCircuit:
        """Builds the full QCNN ansatz with conv and pool layers."""
        qc = QuantumCircuit(self.n_qubits)

        # Helper functions from the seed
        def conv_circuit(params):
            target = QuantumCircuit(2)
            target.rz(-np.pi / 2, 1)
            target.cx(1, 0)
            target.rz(params[0], 0)
            target.ry(params[1], 1)
            target.cx(0, 1)
            target.ry(params[2], 1)
            target.cx(1, 0)
            target.rz(np.pi / 2, 0)
            return target

        def conv_layer(num_qubits, param_prefix):
            qc_layer = QuantumCircuit(num_qubits, name="Convolutional Layer")
            qubits = list(range(num_qubits))
            param_index = 0
            params = ParameterVector(param_prefix, length=num_qubits * 3)
            for q1, q2 in zip(qubits[0::2], qubits[1::2]):
                qc_layer.append(conv_circuit(params[param_index:param_index+3]), [q1, q2])
                qc_layer.barrier()
                param_index += 3
            for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
                qc_layer.append(conv_circuit(params[param_index:param_index+3]), [q1, q2])
                qc_layer.barrier()
                param_index += 3
            return qc_layer

        def pool_circuit(params):
            target = QuantumCircuit(2)
            target.rz(-np.pi / 2, 1)
            target.cx(1, 0)
            target.rz(params[0], 0)
            target.ry(params[1], 1)
            target.cx(0, 1)
            target.ry(params[2], 1)
            return target

        def pool_layer(sources, sinks, param_prefix):
            num_qubits = len(sources) + len(sinks)
            qc_layer = QuantumCircuit(num_qubits, name="Pooling Layer")
            param_index = 0
            params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
            for source, sink in zip(sources, sinks):
                qc_layer.append(pool_circuit(params[param_index:param_index+3]), [source, sink])
                qc_layer.barrier()
                param_index += 3
            return qc_layer

        # Assemble layers
        qc.compose(conv_layer(self.n_qubits, "c1"), inplace=True)
        qc.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), inplace=True)
        qc.compose(conv_layer(self.n_qubits // 2, "c2"), inplace=True)
        qc.compose(pool_layer([0, 1], [2, 3], "p2"), inplace=True)
        qc.compose(conv_layer(self.n_qubits // 4, "c3"), inplace=True)
        qc.compose(pool_layer([0], [1], "p3"), inplace=True)

        return qc

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Evaluate the QCNN and return a tensor of shape (batch, 1)."""
        return self.qnn(inputs)

# Quantum LSTM implementation
class QuantumQLSTM(nn.Module):
    """LSTM cell where gates are realised by small quantum circuits using TorchQuantum."""
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
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
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

# Unified quantum model
class UnifiedQCNNQLSTM(nn.Module):
    """End‑to‑end quantum model that chains QuantumQCNN with QuantumQLSTM."""
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 16,
        tagset_size: int = 0,
        n_qubits_lstm: int = 4,
    ) -> None:
        super().__init__()
        self.qcnn = QuantumQCNN(n_qubits=input_dim)
        self.lstm = QuantumQLSTM(input_dim=1, hidden_dim=hidden_dim, n_qubits=n_qubits_lstm)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """Forward pass for sequence tagging using quantum circuits."""
        seq_len, batch, _ = sentence.size()
        flattened = sentence.view(seq_len * batch, -1)
        qcnn_out = self.qcnn(flattened)  # (batch, 1)
        qcnn_out = qcnn_out.view(seq_len, batch, -1)
        lstm_out, _ = self.lstm(qcnn_out)
        tag_logits = self.hidden2tag(lstm_out.view(seq_len, -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["UnifiedQCNNQLSTM", "QuantumQCNN", "QuantumQLSTM"]
