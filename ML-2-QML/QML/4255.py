import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
from qiskit.circuit.random import random_circuit

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic superposition states for quantum regression."""
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels

class QuantumHybridDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QuanvCircuit:
    def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
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

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        data_np = data.cpu().numpy()
        param_binds = []
        for dat in data_np:
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
        return torch.tensor(counts / (self.shots * self.n_qubits), dtype=torch.float32)

class HybridQuantumRegressionModel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int, conv_kernel: int = 2):
        super().__init__()
        self.num_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

        backend = qiskit.Aer.get_backend("qasm_simulator")
        self.quanv = QuanvCircuit(conv_kernel, backend, shots=100, threshold=0.5)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        conv_out = self.quanv(state_batch)
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        conv_feature = conv_out.unsqueeze(-1)
        combined = features + conv_feature
        return self.head(combined).squeeze(-1)

__all__ = ["HybridQuantumRegressionModel", "QuantumHybridDataset", "generate_superposition_data"]
