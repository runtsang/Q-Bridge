import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit

class QuanvCircuit:
    """Quantum filter circuit inspired by Conv.py QML seed.  It maps a 2×2
    image patch to a 4‑dimensional feature vector consisting of the
    average probability of measuring |1> on each qubit."""
    def __init__(self, filter_size=2, backend=None, shots=100, threshold=127):
        self.n_qubits = filter_size ** 2
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

    def run(self, data):
        """data: 2D array of shape (filter_size, filter_size)."""
        param_binds = []
        for val in data.flatten():
            bind = {self.theta[i]: np.pi if val > self.threshold else 0 for i, val in enumerate(data.flatten())}
            param_binds.append(bind)

        job = qiskit.execute(self._circuit,
                             self.backend,
                             shots=self.shots,
                             parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)

        probs = np.zeros(self.n_qubits)
        for key, count in result.items():
            for i, bit in enumerate(key[::-1]):  # reverse because Qiskit order is reversed
                probs[i] += (int(bit) * count)
        probs = probs / (self.shots * len(param_binds))
        return probs  # shape (n_qubits,)

class BaseQFCModelQuantum(tq.QuantumModule):
    """Original quantum model from QuantumNAT.py."""
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
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

class QFCModel(BaseQFCModelQuantum):
    """Hybrid quantum model that prepends a quanvolution circuit to the
    existing quantum encoder.  The circuit processes each input image
    patch into a 4‑dimensional feature vector, which is then fed into the
    base encoder and variational layer."""
    def __init__(self):
        super().__init__()
        self.quanv = QuanvCircuit(filter_size=2, shots=200)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        patches = x[:, :, :2, :2]  # shape (bsz, 1, 2, 2)
        features = []
        for i in range(bsz):
            patch = patches[i, 0].cpu().numpy()
            feat = self.quanv.run(patch)  # shape (4,)
            features.append(feat)
        features = torch.tensor(features, device=x.device, dtype=torch.float32)  # (bsz, 4)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        self.encoder(qdev, features)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["QFCModel"]
