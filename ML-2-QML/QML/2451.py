import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
import numpy as np
from qiskit.circuit.random import random_circuit

# Quantum filter (QuanvCircuit) from the original QML seed
def Conv():
    class QuanvCircuit:
        """Filter circuit used for quanvolution layers."""
        def __init__(self, kernel_size, backend, shots, threshold):
            self.n_qubits = kernel_size ** 2
            self._circuit = qiskit.QuantumCircuit(self.n_qubits)
            self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
            for i in range(self.n_qubits):
                self._circuit.rx(self.theta[i], i)
            self._circuit.barrier()
            self._circuit += random_circuit(self.n_qubits, 2)
            self._circuit.measure_all()
            self.backend = backend
            self.shots = shots
            self.threshold = threshold

        def run(self, data):
            """Run the quantum circuit on classical data."""
            data = np.reshape(data, (1, self.n_qubits))
            param_binds = []
            for dat in data:
                bind = {}
                for i, val in enumerate(dat):
                    bind[self.theta[i]] = np.pi if val > self.threshold else 0
                param_binds.append(bind)
            job = qiskit.execute(self._circuit,
                                 self.backend,
                                 shots=self.shots,
                                 parameter_binds=param_binds)
            result = job.result().get_counts(self._circuit)
            counts = 0
            for key, val in result.items():
                ones = sum(int(bit) for bit in key)
                counts += ones * val
            return counts / (self.shots * self.n_qubits)

    backend = qiskit.Aer.get_backend("qasm_simulator")
    filter_size = 2
    circuit = QuanvCircuit(filter_size, backend, shots=100, threshold=127)
    return circuit

class HybridNATModel(tq.QuantumModule):
    """
    Hybrid quantum‑classical model that merges a convolutional feature extractor,
    a quantum filter, and a variational quantum layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.quantum_filter = Conv()  # quantum filter circuit

        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)

        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Classical variational layer to emulate the quantum block
        self.q_layer_classical = nn.Sequential(
            nn.Linear(16 * 7 * 7, 4),
            nn.Tanh()
        )

        # Final fully‑connected head
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7 + 1 + 4 + 4, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)

    class QLayer(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]

        # Classical feature extractor
        features = self.features(x).view(bsz, -1)  # (bsz, 784)

        # Quantum filter output (scalar per image)
        filter_out = torch.tensor(
            [self.quantum_filter.run(x[i, 0].cpu().numpy()) for i in range(bsz)],
            device=x.device
        ).unsqueeze(1)  # (bsz, 1)

        # Quantum device and variational circuit
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        q_out = self.measure(qdev)  # (bsz, 4)

        # Classical variational layer (optional)
        q_out_classical = self.q_layer_classical(features)  # (bsz, 4)

        # Concatenate all signals
        concat = torch.cat([features, filter_out, q_out, q_out_classical], dim=1)

        out = self.fc(concat)
        return self.norm(out)

__all__ = ["HybridNATModel"]
