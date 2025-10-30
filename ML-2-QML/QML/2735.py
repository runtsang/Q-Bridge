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
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

# ----- QCNN quantum feature extractor -----
def conv_circuit(params):
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

def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits)
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc.compose(conv_circuit(params[param_index:param_index + 3]), [q1, q2], inplace=True)
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc.compose(conv_circuit(params[param_index:param_index + 3]), [q1, q2], inplace=True)
        param_index += 3
    return qc

def pool_circuit(params):
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for src, sink in zip(sources, sinks):
        qc.compose(pool_circuit(params[param_index:param_index + 3]), [src, sink], inplace=True)
        param_index += 3
    return qc

class QCNNQuantum(nn.Module):
    """Quantum QCNN feature extractor using EstimatorQNN."""
    def __init__(self):
        super().__init__()
        self.estimator = Estimator()
        self.feature_map = ZFeatureMap(8)
        self.ansatz = self._build_ansatz()
        self.qnn = EstimatorQNN(
            circuit=self.ansatz,
            observables=SparsePauliOp.from_list([("Z" + "I" * 7, 1)]),
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    def _build_ansatz(self):
        ansatz = QuantumCircuit(8)
        ansatz.compose(conv_layer(8, "c1"), inplace=True)
        ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), inplace=True)
        ansatz.compose(conv_layer(4, "c2"), inplace=True)
        ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), inplace=True)
        ansatz.compose(conv_layer(2, "c3"), inplace=True)
        ansatz.compose(pool_layer([0], [1], "p3"), inplace=True)
        return ansatz.decompose()

    def forward(self, x):
        # x: (batch, seq, feat)
        batch, seq, feat = x.shape
        flat = x.view(batch * seq, feat)
        outputs = self.qnn(flat)
        return outputs.view(batch, seq, -1)

# ----- Quantum LSTM gates -----
class QLayer(tq.QuantumModule):
    def __init__(self, n_wires):
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

    def forward(self, x):
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

# ----- Hybrid Quantum LSTM -----
class HybridQLSTM(nn.Module):
    """Hybrid quantum‑classical LSTM with QCNN feature extractor."""
    def __init__(self, input_dim, hidden_dim, tagset_size, n_qubits, use_qcnn=True):
        super().__init__()
        self.n_qubits = n_qubits
        self.hidden_dim = hidden_dim
        self.feature_extractor = QCNNQuantum() if use_qcnn else None
        self.forget = QLayer(n_qubits)
        self.input = QLayer(n_qubits)
        self.update = QLayer(n_qubits)
        self.output = QLayer(n_qubits)
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, inputs):
        # inputs: (batch, seq, feat)
        batch, seq, feat = inputs.shape
        if self.feature_extractor is not None:
            features = self.feature_extractor(inputs)  # (batch, seq, 1)
            inputs = features
        hx = torch.zeros(batch, self.hidden_dim, device=inputs.device)
        cx = torch.zeros(batch, self.hidden_dim, device=inputs.device)
        outputs = []
        for t in range(seq):
            x = inputs[:, t, :]
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        logits = self.hidden2tag(outputs)
        return F.log_softmax(logits, dim=-1)

# Alias for backward compatibility
QLSTM = HybridQLSTM

__all__ = ["HybridQLSTM", "QCNNQuantum", "QLSTM"]
