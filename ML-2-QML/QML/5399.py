import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer

class SamplerQNNQuantum:
    """Parameterized Qiskit circuit that emulates the classical sampler."""
    def __init__(self) -> None:
        self.circuit = QuantumCircuit(2)
        self.inputs = [QuantumCircuit.Parameter(f"input_{i}") for i in range(2)]
        self.weights = [QuantumCircuit.Parameter(f"weight_{i}") for i in range(4)]
        self.circuit.ry(self.inputs[0], 0)
        self.circuit.ry(self.inputs[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weights[0], 0)
        self.circuit.ry(self.weights[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weights[2], 0)
        self.circuit.ry(self.weights[3], 1)
        self.backend = Aer.get_backend('qasm_simulator')
    def sample(self, input_vals, weight_vals, shots=1024) -> dict:
        param_dict = {
            self.inputs[0]: input_vals[0],
            self.inputs[1]: input_vals[1],
            self.weights[0]: weight_vals[0],
            self.weights[1]: weight_vals[1],
            self.weights[2]: weight_vals[2],
            self.weights[3]: weight_vals[3],
        }
        job = execute(self.circuit, self.backend, parameter_binds=[param_dict], shots=shots)
        return job.result().get_counts(self.circuit)

class QFCModelQuantum(tq.QuantumModule):
    """Quantum‑equivalent of the CNN‑FC model."""
    class _QLayer(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.random = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)
        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=3)
            self.crx(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self._QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

class QuanvolutionFilterQuantum(tq.QuantumModule):
    """Two‑qubit quantum kernel applied to 2×2 patches."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder([
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ])
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        dev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [x[:, r, c], x[:, r, c+1], x[:, r+1, c], x[:, r+1, c+1]],
                    dim=1
                )
                self.encoder(dev, data)
                self.q_layer(dev)
                patches.append(self.measure(dev).view(bsz, 4))
        return torch.cat(patches, dim=1)

class QuantumSelfAttention:
    """Self‑attention block implemented with Qiskit."""
    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = Aer.get_backend("qasm_simulator")
    def _build(self, rot: np.ndarray, ent: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            qc.rx(rot[3*i], i)
            qc.ry(rot[3*i+1], i)
            qc.rz(rot[3*i+2], i)
        for i in range(self.n_qubits-1):
            qc.crx(ent[i], i, i+1)
        qc.measure(self.qr, self.cr)
        return qc
    def run(self, rot: np.ndarray, ent: np.ndarray, shots: int = 1024) -> dict:
        qc = self._build(rot, ent)
        job = execute(qc, self.backend, shots=shots)
        return job.result().get_counts(qc)

class HybridSamplerQNNQuantum(tq.QuantumModule):
    """Quantum‑centric hybrid that mirrors HybridSamplerQNN."""
    def __init__(self) -> None:
        super().__init__()
        self.sampler = SamplerQNNQuantum()
        self.qfc = QFCModelQuantum()
        self.filter = QuanvolutionFilterQuantum()
        self.attn = QuantumSelfAttention()
    def forward(self, x: torch.Tensor) -> dict:
        # x expected to be (batch, 2) for the sampler
        input_vals = [x[0,0].item(), x[0,1].item()]
        weight_vals = [0,0,0,0]
        samp_counts = self.sampler.sample(input_vals, weight_vals, shots=512)
        # Dummy image for the remaining modules
        img = torch.zeros(x.size(0), 1, 28, 28, device=x.device)
        img[:, 0, 0, 0] = x[:, 0]
        img[:, 0, 0, 1] = x[:, 1]
        qfc_out = self.qfc(img)
        quanv_out = self.filter(img)
        # Concatenate tensors for attention
        feat = torch.cat([qfc_out, quanv_out], dim=1)
        rot = np.random.randn(12)
        ent = np.random.randn(3)
        att_counts = self.attn.run(rot, ent, shots=512)
        return {"sample_counts": samp_counts, "qfc": qfc_out, "quanv": quanv_out, "attention_counts": att_counts}
