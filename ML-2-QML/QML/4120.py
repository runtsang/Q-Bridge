"""Quantum‑enhanced QCNN with LSTM, leveraging Qiskit for quantum convolution and pooling, and TorchQuantum for a quantum LSTM."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
import torchquantum as tq
import torchquantum.functional as tqf

# Quantum convolution and pooling primitives (from QCNN.py)
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
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3
    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc

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
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
        qc.barrier()
        param_index += 3
    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc

# Quantum LSTM implementation (from QLSTM.py)
class QLSTM(nn.Module):
    """LSTM cell where gates are realized by small quantum circuits."""

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

    def forward(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None = None):
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

class QCNNHybrid(nn.Module):
    """Quantum‑enhanced QCNN that fuses quantum convolution, pooling, and a quantum LSTM."""

    def __init__(
        self,
        input_dim: int = 8,
        lstm_hidden_dim: int = 16,
        n_qubits: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim

        # Feature map
        self.feature_map = ZFeatureMap(input_dim)

        # Ansatz construction (three conv‑pool stages)
        ansatz = QuantumCircuit(input_dim)
        ansatz.compose(conv_layer(input_dim, "c1"), range(input_dim), inplace=True)
        ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), range(input_dim), inplace=True)
        ansatz.compose(conv_layer(input_dim // 2, "c2"), range(input_dim // 2), inplace=True)
        ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), range(input_dim // 2), inplace=True)
        ansatz.compose(conv_layer(input_dim // 4, "c3"), range(input_dim // 4), inplace=True)
        ansatz.compose(pool_layer([0], [1], "p3"), range(input_dim // 4), inplace=True)

        # Observable
        observable = SparsePauliOp.from_list([("Z" + "I" * (input_dim - 1), 1)])

        # QNN
        self.qnn = EstimatorQNN(
            circuit=ansatz.decompose(),
            observables=observable,
            input_params=self.feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=Estimator(),
        )

        # Quantum LSTM
        self.lstm = QLSTM(input_dim=input_dim, hidden_dim=lstm_hidden_dim, n_qubits=n_qubits)

        # Classification head
        self.head = nn.Linear(lstm_hidden_dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Quantum feature extraction
        qfeat = self.qnn(inputs)  # (batch, input_dim)
        seq = qfeat.unsqueeze(1)  # (batch, seq_len=1, input_dim)
        lstm_out, _ = self.lstm(seq)
        out = self.head(lstm_out[:, -1, :])
        return torch.sigmoid(out)

def QCNNHybridFactory() -> QCNNHybrid:
    """Return a ready‑to‑train QCNNHybrid instance."""
    return QCNNHybrid()

__all__ = ["QCNNHybrid", "QCNNHybridFactory"]
