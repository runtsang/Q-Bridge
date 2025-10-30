import torch
import torch.nn as nn
import torchquantum as tq
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.circuit.library import ZFeatureMap

class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Quantum two‑qubit kernel applied to 2×2 patches."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [0], "func": "ry", "wires": [0]},
             {"input_idx": [1], "func": "ry", "wires": [1]},
             {"input_idx": [2], "func": "ry", "wires": [2]},
             {"input_idx": [3], "func": "ry", "wires": [3]}]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [x[:, r, c], x[:, r, c+1], x[:, r+1, c], x[:, r+1, c+1]],
                    dim=1
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        sub = QuantumCircuit(2)
        sub.rz(-np.pi/2, 1)
        sub.cx(1,0)
        sub.rz(params[param_index], 0)
        sub.ry(params[param_index+1], 1)
        sub.cx(0,1)
        sub.ry(params[param_index+2], 1)
        sub.cx(1,0)
        sub.rz(np.pi/2, 0)
        qc.append(sub, [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2]+[0]):
        sub = QuantumCircuit(2)
        sub.rz(-np.pi/2, 1)
        sub.cx(1,0)
        sub.rz(params[param_index], 0)
        sub.ry(params[param_index+1], 1)
        sub.cx(0,1)
        sub.ry(params[param_index+2], 1)
        sub.cx(1,0)
        sub.rz(np.pi/2, 0)
        qc.append(sub, [q1, q2])
        qc.barrier()
        param_index += 3
    return qc

def pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits//2 * 3)
    for source, sink in zip(sources, sinks):
        sub = QuantumCircuit(2)
        sub.rz(-np.pi/2, 1)
        sub.cx(1,0)
        sub.rz(params[param_index], 0)
        sub.ry(params[param_index+1], 1)
        sub.cx(0,1)
        sub.ry(params[param_index+2], 1)
        qc.append(sub, [source, sink])
        qc.barrier()
        param_index += 3
    return qc

class QuanvolutionHybridQML(nn.Module):
    """Hybrid quantum model: quanvolution → self‑attention → QCNN ansatz."""
    def __init__(self):
        super().__init__()
        self.qfilter = QuantumQuanvolutionFilter()
        self.qnn = self._build_qnn()

    def _build_qnn(self) -> EstimatorQNN:
        fm = ZFeatureMap(4)
        sa_params = ParameterVector("sa", length=12)  # 4 qubits × 3 rotations
        sa_circ = QuantumCircuit(4)
        for i in range(4):
            sa_circ.rx(sa_params[3*i], i)
            sa_circ.ry(sa_params[3*i+1], i)
            sa_circ.rz(sa_params[3*i+2], i)
        for i in range(3):
            sa_circ.cx(i, i+1)
        ansatz_circ = conv_layer(4, "c")
        circuit = QuantumCircuit(4)
        circuit.compose(fm, range(4), inplace=True)
        circuit.compose(sa_circ, range(4), inplace=True)
        circuit.compose(ansatz_circ, range(4), inplace=True)
        estimator = Estimator()
        observable = SparsePauliOp.from_list([("Z"*4, 1)])
        weight_params = list(sa_params) + list(ansatz_circ.parameters)
        return EstimatorQNN(
            circuit=circuit,
            observables=observable,
            input_params=fm.parameters,
            weight_params=weight_params,
            estimator=estimator,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract quantum features from the image
        features = self.qfilter(x)
        # For demonstration, we take the first 4 features per sample to feed the QNN
        batched = features[:, :4]
        # Evaluate the quantum neural network
        out = self.qnn(batched)
        return torch.log_softmax(out, dim=-1)

__all__ = ["QuantumQuanvolutionFilter", "QuanvolutionHybridQML"]
