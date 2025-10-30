import numpy as np
import torch
from torch import nn
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import RealAmplitudes
from qiskit_aer import AerSimulator
from qiskit.circuit.random import random_circuit

class QuanvCircuit:
    """Quantum convolutional filter that maps a 2D array to a probability."""
    def __init__(self, kernel_size: int, backend=None, shots: int = 100, threshold: float = 127):
        self.n_qubits = kernel_size ** 2
        self.backend = backend or Aer.get_backend('qasm_simulator')
        self.shots = shots
        self.threshold = threshold
        self.circuit = QuantumCircuit(self.n_qubits)
        self.theta = [QuantumCircuit.Parameter(f'theta{i}') for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)
        job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self.circuit)
        total = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            total += ones * val
        return total / (self.shots * self.n_qubits)

class QuantumAutoencoderCircuit:
    """Variational auto‑encoder circuit with swap‑test fidelity."""
    def __init__(self, latent_dim: int):
        self.latent_dim = latent_dim
        self.circuit = QuantumCircuit(latent_dim + 1)  # +1 auxiliary qubit
        ansatz = RealAmplitudes(latent_dim, reps=5)
        self.circuit.append(ansatz, range(latent_dim))
        self.circuit.h(0)
        for i in range(latent_dim):
            self.circuit.cswap(0, i, latent_dim + i)
        self.circuit.h(0)
        self.circuit.measure(0, 0)
        self.simulator = AerSimulator()

    def run(self, params: np.ndarray) -> float:
        bound = {self.circuit.parameters[i]: float(params[i]) for i in range(len(params))}
        circuit = self.circuit.copy()
        circuit.bind_parameters(bound)
        job = self.simulator.run(circuit, shots=100)
        result = job.result()
        counts = result.get_counts()
        total = 0
        for key, value in counts.items():
            total += int(key) * value
        return total / (100 * 1)

class HybridAutoencoder(nn.Module):
    """Hybrid classical‑quantum auto‑encoder with convolutional pre‑processing."""
    def __init__(self, latent_dim: int, conv_kernel: int = 2):
        super().__init__()
        self.conv = QuanvCircuit(conv_kernel)
        self.quantum_autoencoder = QuantumAutoencoderCircuit(latent_dim)
        self.decoder = nn.Linear(latent_dim, latent_dim)

    def forward(self, x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        conv_out = self.conv.run(x)
        fidelity = self.quantum_autoencoder.run(np.array([conv_out]))
        latent = torch.tensor([conv_out], dtype=torch.float32)
        recon = self.decoder(latent)
        return recon, torch.tensor(fidelity, dtype=torch.float32)

def train_hybrid_autoencoder_qml(
    model: HybridAutoencoder,
    data: np.ndarray,
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
) -> list[float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()
    history: list[float] = []
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        recon, fidelity = model.forward(data)
        loss = mse(recon, torch.tensor(data, dtype=torch.float32)) + (1 - fidelity)
        loss.backward()
        opt.step()
        history.append(loss.item())
    return history

__all__ = ['HybridAutoencoder', 'train_hybrid_autoencoder_qml']
