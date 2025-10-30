import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import qiskit

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Produce superposition states |ψ⟩ = cos(θ)|0…0⟩ + e^{iϕ} sin(θ)|1…1⟩
    and target values y = sin(2θ) cos(ϕ).  Mirrors the classical dataset but
    outputs complex amplitudes for a quantum device.
    """
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
    return states, labels.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Quantum version of the dataset.  Each sample is a complex state vector
    and a real target value.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QuantumFullyConnectedLayer:
    """
    Parameterised quantum circuit that emulates a fully‑connected layer.
    Each input feature is treated as a rotation angle applied to a qubit.
    """
    def __init__(self, n_qubits: int, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> qiskit.QuantumCircuit:
        qc = qiskit.QuantumCircuit(self.n_qubits)
        qc.h(range(self.n_qubits))
        theta = qiskit.circuit.Parameter('theta')
        qc.ry(theta, range(self.n_qubits))
        qc.measure(range(self.n_qubits), range(self.n_qubits))
        return qc

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for a batch of rotation angles and return
        the vector of Pauli‑Z expectations for each qubit.
        """
        param_binds = [{self.circuit.parameters[0]: theta} for theta in thetas]
        job = qiskit.execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        expectations = np.zeros(self.n_qubits)
        for state, count in counts.items():
            prob = count / self.shots
            for i in range(self.n_qubits):
                # state string is MSB first
                if state[self.n_qubits - 1 - i] == '1':
                    expectations[i] -= prob
                else:
                    expectations[i] += prob
        return expectations

class HybridFCLRegression:
    """
    Quantum‑classical hybrid regression model.  The quantum layer produces
    a feature vector of Pauli‑Z expectations; a classical linear head
    maps this to a scalar prediction.
    """
    def __init__(self, n_qubits: int, hidden_dim: int = 32, shots: int = 1024):
        self.q_layer = QuantumFullyConnectedLayer(n_qubits, shots)
        self.head = nn.Linear(n_qubits, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: (batch, n_qubits) → (batch, 1)
        """
        batch = x.shape[0]
        expectations = []
        for i in range(batch):
            thetas = x[i].detach().cpu().numpy()
            exp = self.q_layer.run(thetas)
            expectations.append(exp)
        expectations = torch.tensor(expectations, dtype=torch.float32, device=x.device)
        return self.head(expectations).squeeze(-1)

__all__ = ["HybridFCLRegression", "RegressionDataset", "generate_superposition_data"]
