import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from qiskit import Aer
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
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

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QuantumSelfAttention:
    """Quantum circuit implementing a self‑attention style block."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.backend = Aer.get_backend("statevector_simulator")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray, state: np.ndarray) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits, "q")
        qc = QuantumCircuit(qr)
        # Encode the input state as amplitudes
        qc.initialize(state, qr)
        # Parameterised rotations
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        # Entangling CRX gates
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)
        return qc

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, state: np.ndarray) -> np.ndarray:
        qc = self._build_circuit(rotation_params, entangle_params, state)
        result = self.backend.run(qc).result()
        statevector = result.get_statevector(qc)
        probs = np.abs(statevector)**2
        # Compute Pauli‑Z expectation for each qubit
        expectations = []
        for q in range(self.n_qubits):
            mask = 1 << (self.n_qubits - q - 1)
            exp = 0.0
            for idx, prob in enumerate(probs):
                bit = 1 if (idx & mask) else 0
                exp += ((-1) ** bit) * prob
            expectations.append(exp)
        return np.array(expectations, dtype=np.float32)

class QModel(nn.Module):
    """Hybrid quantum regression model: quantum self‑attention + linear head."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.num_wires = num_wires
        self.attention = QuantumSelfAttention(num_wires)
        # Trainable parameters for the quantum block
        self.rotation_params = nn.Parameter(torch.randn(3 * num_wires))
        self.entangle_params = nn.Parameter(torch.randn(num_wires - 1))
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        batch_features = []
        for state in state_batch:
            # Convert complex tensor to numpy array of amplitudes
            amp = state.detach().cpu().numpy()
            features = self.attention.run(
                self.rotation_params.detach().cpu().numpy(),
                self.entangle_params.detach().cpu().numpy(),
                amp,
            )
            batch_features.append(features)
        features_tensor = torch.tensor(batch_features, dtype=torch.float32, device=state_batch.device)
        return self.head(features_tensor).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
