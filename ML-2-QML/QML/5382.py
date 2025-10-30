import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import transpile, assemble
from qiskit.providers.aer import AerSimulator


class QuantumFraudCircuit:
    """Two‑qubit parameterised circuit that maps a real value to a measurement probability."""
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.ry(self.theta, 0)
        self.circuit.ry(2 * self.theta, 1)
        self.circuit.barrier()
        self.circuit.cx(0, 1)
        self.circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        job = self.backend.run(
            assemble(
                compiled,
                shots=self.shots,
                parameter_binds=[{self.theta: theta} for theta in thetas],
            )
        )
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(int)
            probs = counts / self.shots
            z_vals = 1 - 2 * states
            return np.sum(z_vals * probs)
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])


class QuantumFraudLayer(nn.Module):
    """Hybrid layer that forwards activations through the quantum circuit."""
    def __init__(self, n_qubits: int = 2, shots: int = 200) -> None:
        super().__init__()
        backend = AerSimulator()
        self.quantum_circuit = QuantumFraudCircuit(n_qubits, backend, shots)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch = inputs.shape[0]
        thetas = inputs.detach().cpu().numpy().reshape(-1)
        expectations = self.quantum_circuit.run(thetas)
        probs = (expectations + 1) / 2
        return torch.tensor(probs, device=inputs.device).unsqueeze(-1)


class HybridQCNet(nn.Module):
    """Quantum‑enhanced binary classifier."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        self.quantum_head = QuantumFraudLayer(n_qubits=2, shots=200)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.quantum_head(x)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["QuantumFraudCircuit", "QuantumFraudLayer", "HybridQCNet"]
