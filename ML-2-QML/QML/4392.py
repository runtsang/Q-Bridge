import torch
import torch.nn as nn
import numpy as np
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

class QuanvolutionFilter(tq.QuantumModule):
    """Random two‑qubit quantum kernel applied to 2×2 image patches."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class QLayer(tq.QuantumModule):
    """Variational sub‑module inspired by Quantum‑NAT."""
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

class QuantumSelfAttention:
    """Quantum‑style self‑attention block built with Qiskit."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        circuit = QuantumCircuit(qr, cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], qr[i])
            circuit.ry(rotation_params[3 * i + 1], qr[i])
            circuit.rz(rotation_params[3 * i + 2], qr[i])
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], qr[i], qr[i + 1])
        circuit.measure(qr, cr)
        return circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, self.backend, shots=shots)
        result = job.result().get_counts(circuit)
        probs = np.array(list(result.values())) / shots
        states = np.array([int(k, 2) for k in result.keys()])  # binary to int
        expectation = np.sum(states * probs)
        return torch.tensor([expectation], dtype=torch.float32)

class HybridFCL(tq.QuantumModule):
    """
    Quantum‑centric hybrid model that replaces the classical blocks in
    HybridFCL with variational circuits and quantum kernels.
    """
    def __init__(self, n_features: int = 1, n_classes: int = 10) -> None:
        super().__init__()
        # Quantum encoder (Quanvolution)
        self.encoder = QuanvolutionFilter()
        # Quantum fully‑connected layer
        self.q_fc = QLayer()
        # Quantum self‑attention
        self.attention = QuantumSelfAttention(n_qubits=4)
        # Classical head
        self.classifier = nn.Linear(n_features, n_classes)
        self.norm = nn.BatchNorm1d(n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        # Quantum encoder
        features = self.encoder(x)  # shape (bsz, n_features)
        # Quantum fully‑connected layer
        qdev = tq.QuantumDevice(n_wires=4, bsz=bsz, device=x.device, record_op=True)
        self.q_fc(qdev)
        fc_out = self.q_fc.measure(qdev)  # (bsz, n_features)
        # Quantum self‑attention (simple expectation)
        rotation_params = np.random.rand(12)
        entangle_params = np.random.rand(3)
        attn_out = self.attention.run(rotation_params, entangle_params, shots=1024)
        attn_out = attn_out.expand(bsz, -1).to(x.device)
        # Combine
        combined = fc_out + attn_out
        combined = self.norm(combined)
        logits = self.classifier(combined)
        return logits

    def run(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

__all__ = ["HybridFCL"]
