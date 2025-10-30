import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

def build_qcnn(num_qubits: int = 8, n_layers: int = 3) -> EstimatorQNN:
    """
    Builds a QCNN ansatz with convolution and pooling layers
    and returns an EstimatorQNN that can be used as a PyTorch module.
    """
    # Feature map: simple Z‑feature map
    feature_map = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        feature_map.rz(ParameterVector('x', length=num_qubits)[i], i)

    # Convolution layer: entangle pairs with RZ and CX
    def conv_layer(params, qubits):
        qc = QuantumCircuit(len(qubits))
        for i in range(0, len(qubits)-1, 2):
            qc.cx(qubits[i], qubits[i+1])
            qc.rz(params[i], qubits[i+1])
            qc.ry(params[i+1], qubits[i])
            qc.cx(qubits[i+1], qubits[i])
        return qc

    # Pooling layer: entangle pairs and discard one qubit
    def pool_layer(params, qubits):
        qc = QuantumCircuit(len(qubits))
        for i in range(0, len(qubits)-1, 2):
            qc.cx(qubits[i], qubits[i+1])
            qc.rz(params[i], qubits[i+1])
            qc.ry(params[i+1], qubits[i])
        return qc

    ansatz = QuantumCircuit(num_qubits)

    for layer in range(n_layers):
        conv_params = ParameterVector(f'c{layer}', length=num_qubits//2*2)
        pool_params = ParameterVector(f'p{layer}', length=num_qubits//2*2)
        ansatz.compose(conv_layer(conv_params, range(num_qubits)), inplace=True)
        ansatz.compose(pool_layer(pool_params, range(num_qubits)), inplace=True)

    observable = SparsePauliOp.from_list([('Z'*num_qubits, 1.0)])
    estimator = Estimator()
    qnn = EstimatorQNN(
        circuit=ansatz,
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator
    )
    return qnn

class QLayer(tq.QuantumModule):
    """
    Simple quantum layer that encodes a real vector via RX gates,
    applies a trainable RZ sequence and a chain of CNOTs, and measures
    all qubits in the Z basis.
    """
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.params = nn.ModuleList([tq.RZ(has_params=True, trainable=True) for _ in range(n_qubits)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for i, gate in enumerate(self.params):
            gate(qdev, wires=[i])
        for i in range(self.n_qubits - 1):
            tqf.cnot(qdev, wires=[i, i+1])
        return self.measure(qdev)

def build_qlstm(input_dim: int, hidden_dim: int, n_qubits: int):
    """
    Returns a quantum‑enhanced LSTM module that uses QLayer for each gate.
    """
    class QLSTMCell(nn.Module):
        def __init__(self):
            super().__init__()
            self.forget = QLayer(n_qubits)
            self.input = QLayer(n_qubits)
            self.update = QLayer(n_qubits)
            self.output = QLayer(n_qubits)

            self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        def forward(self, x, hx, cx):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            return hx, cx

    class QuantumLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.cell = QLSTMCell()

        def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] = None):
            batch = inputs.size(1)
            if states is None:
                hx = torch.zeros(batch, hidden_dim, device=inputs.device)
                cx = torch.zeros(batch, hidden_dim, device=inputs.device)
            else:
                hx, cx = states
            outputs = []
            for t in range(inputs.size(0)):
                hx, cx = self.cell(inputs[t], hx, cx)
                outputs.append(hx.unsqueeze(0))
            return torch.cat(outputs, dim=0), (hx, cx)

    return QuantumLSTM()
