import numpy as np
import torch
import torch.nn as nn
import qiskit
from qiskit.circuit.random import random_circuit

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

class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class ConvGen622(nn.Module):
    """
    Hybrid classical/quantum convolutional filter with shared regression head.
    The filter can be toggled between a classical Conv2d and a variational quantum circuit.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0,
                 use_quantum: bool = True, qbackend: str = "qasm_simulator",
                 shots: int = 100):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.use_quantum = use_quantum
        self.qbackend = qbackend
        self.shots = shots
        if use_quantum:
            self.filter_circuit = self._build_quantum_filter()
        else:
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        # shared regression head
        self.head = nn.Linear(1, 1)

    def _build_quantum_filter(self):
        n_qubits = self.kernel_size ** 2
        circuit = qiskit.QuantumCircuit(n_qubits)
        theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            circuit.rx(theta[i], i)
        circuit.barrier()
        circuit += random_circuit(n_qubits, 2)
        circuit.measure_all()
        backend = qiskit.Aer.get_backend(self.qbackend)
        return {"circuit": circuit, "backend": backend, "shots": self.shots, "theta": theta, "threshold": self.threshold}

    def _run_filter(self, data: np.ndarray) -> float:
        data = np.reshape(data, (1, self.kernel_size ** 2))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.filter_circuit["theta"][i]] = np.pi if val > self.filter_circuit["threshold"] else 0
            param_binds.append(bind)
        job = qiskit.execute(self.filter_circuit["circuit"],
                             self.filter_circuit["backend"],
                             shots=self.filter_circuit["shots"],
                             parameter_binds=param_binds)
        result = job.result().get_counts(self.filter_circuit["circuit"])
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.filter_circuit["shots"] * self.kernel_size ** 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, H, W)
        if self.use_quantum:
            # run quantum filter on each sample
            feats = []
            for i in range(x.size(0)):
                data = x[i].cpu().numpy().reshape(self.kernel_size, self.kernel_size)
                feats.append(self._run_filter(data))
            feats = torch.tensor(feats, dtype=torch.float32, device=x.device).unsqueeze(-1)
        else:
            logits = self.conv(x)
            activations = torch.sigmoid(logits - self.threshold)
            feats = activations.view(activations.size(0), -1).mean(dim=1, keepdim=True)
        return self.head(feats).squeeze(-1)

    def run(self, data: np.ndarray) -> float:
        """Run the quantum filter on a single 2D array."""
        return self._run_filter(data)

def Conv(kernel_size: int = 2, threshold: float = 0.0) -> ConvGen622:
    """Backward‑compatible factory returning a quantum ConvGen622 instance."""
    return ConvGen622(kernel_size=kernel_size, threshold=threshold, use_quantum=True)

__all__ = ["ConvGen622", "RegressionDataset", "generate_superposition_data", "Conv"]
