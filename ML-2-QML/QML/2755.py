import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import qiskit
from qiskit import QuantumCircuit, Aer, execute

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate quantum superposition states and a regression target."""
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

class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QuantumSelfAttention(tq.QuantumModule):
    """Quantum self‑attention block built with Qiskit."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.num_wires = num_wires
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        qr = qiskit.QuantumRegister(self.num_wires, "q")
        cr = qiskit.ClassicalRegister(self.num_wires, "c")
        circuit = QuantumCircuit(qr, cr)
        for i in range(self.num_wires):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.num_wires - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(qr, cr)
        return circuit

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        batch = qdev.bsz
        # Random parameters per sample
        rot_params = torch.randn(batch, 3 * self.num_wires)
        ent_params = torch.randn(batch, self.num_wires - 1)
        features = []
        for i in range(batch):
            circ = self._build_circuit(rot_params[i].cpu().numpy(), ent_params[i].cpu().numpy())
            job = execute(circ, self.backend, shots=1024)
            counts = job.result().get_counts(circ)
            # Convert counts to a simple feature vector (placeholder)
            vec = np.zeros(self.num_wires, dtype=np.float32)
            for key, val in counts.items():
                for idx, bit in enumerate(reversed(key)):
                    vec[idx] += (1 if bit == "1" else -1) * val
            features.append(vec)
        return torch.tensor(features, dtype=torch.float32, device=qdev.device)

class QuantumRegressionWithAttention(tq.QuantumModule):
    """Quantum regression model with an embedded self‑attention block."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.attention = QuantumSelfAttention(num_wires)
        self.q_layer = tq.RandomLayer(n_ops=20, wires=list(range(num_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.encoder.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        attn_feat = self.attention(qdev)
        # For simplicity, we use only the attention features as input to the head
        return self.head(attn_feat).squeeze(-1)

__all__ = ["QuantumRegressionWithAttention", "RegressionDataset", "generate_superposition_data"]
