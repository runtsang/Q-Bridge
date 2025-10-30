import torch
import torch.nn as nn
from typing import Tuple
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
import torchquantum as tq
import torchquantum.functional as tqf

# ------------------------------------------------------------------
# QCNN ansatz construction
# ------------------------------------------------------------------
def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi/2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(np.pi/2, 0)
    return qc

def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        sub = conv_circuit(params[idx:idx+3])
        qc.compose(sub, [q1, q2], inplace=True)
        qc.barrier()
        idx += 3
    if num_qubits % 2 == 1:
        sub = conv_circuit(params[idx:idx+3])
        qc.compose(sub, [num_qubits-1, 0], inplace=True)
        qc.barrier()
    return qc

def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi/2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def pool_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=(num_qubits//2)*3)
    idx = 0
    for src, sink in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        sub = pool_circuit(params[idx:idx+3])
        qc.compose(sub, [src, sink], inplace=True)
        qc.barrier()
        idx += 3
    return qc

def build_qcnn_ansatz() -> QuantumCircuit:
    feature_map = ZFeatureMap(8)
    ansatz = QuantumCircuit(8)
    ansatz.compose(conv_layer(8, "c1"), inplace=True)
    ansatz.compose(pool_layer(8, "p1"), inplace=True)
    ansatz.compose(conv_layer(4, "c2"), inplace=True)
    ansatz.compose(pool_layer(4, "p2"), inplace=True)
    ansatz.compose(conv_layer(2, "c3"), inplace=True)
    ansatz.compose(pool_layer(2, "p3"), inplace=True)
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)
    return circuit

# ------------------------------------------------------------------
# Quantum LSTM cell using torchquantum
# ------------------------------------------------------------------
class QLSTMQuantum(tq.QuantumModule):
    """Quantum LSTM cell with parameterized gates for gates."""
    class QGate(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for i, gate in enumerate(self.params):
                gate(qdev, wires=i)
            for i in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[i, i+1])
            tqf.cnot(qdev, wires=[self.n_wires-1, 0])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget = self.QGate(n_qubits)
        self.input = self.QGate(n_qubits)
        self.update = self.QGate(n_qubits)
        self.output = self.QGate(n_qubits)
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None):
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
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch_size, self.hidden_dim, device=device), torch.zeros(batch_size, self.hidden_dim, device=device)

# ------------------------------------------------------------------
# Hybrid QCNN + Quantum LSTM
# ------------------------------------------------------------------
class HybridQCNNQLSTMQuantum(nn.Module):
    """Hybrid quantum model combining QCNN feature extraction with a quantum LSTM."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 4):
        super().__init__()
        self.qcnn_circuit = build_qcnn_ansatz()
        self.qcnn_estimator = Estimator()
        self.qcnn_qnn = EstimatorQNN(
            circuit=self.qcnn_circuit,
            observables=SparsePauliOp.from_list([("Z" + "I"*7, 1)]),
            input_params=self.qcnn_circuit.parameters,
            estimator=self.qcnn_estimator,
        )
        self.qlstm = QLSTMQuantum(input_dim=1, hidden_dim=hidden_dim, n_qubits=n_qubits)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, features) where features == 8.

        Returns
        -------
        qcnn_out : torch.Tensor
            QCNN feature map output for each timestep.
        logits : torch.Tensor
            Final classification logits from the quantum LSTM.
        """
        batch, seq_len, _ = x.shape
        flat = x.reshape(batch * seq_len, -1)
        qcnn_out = self.qcnn_qnn(flat).reshape(batch, seq_len, -1)
        lstm_in = qcnn_out.permute(1, 0, 2)
        lstm_out, _ = self.qlstm(lstm_in)
        last_hidden = lstm_out[-1]
        logits = self.head(last_hidden)
        return qcnn_out, logits

__all__ = ["HybridQCNNQLSTMQuantum", "QLSTMQuantum"]
