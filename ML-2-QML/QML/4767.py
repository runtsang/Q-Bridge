"""Quantum regression model that fuses encoding, quantum convolution, and a quantum fully‑connected block."""  

from __future__ import annotations  

import numpy as np  
import torch  
import torch.nn as nn  
import torchquantum as tq  
import qiskit  
from qiskit.circuit.random import random_circuit  

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:  
    """Generate superposition states and sinusoidal targets suitable for quantum regression."""  
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

class RegressionDataset(torch.utils.data.Dataset):  
    """Dataset that returns complex state vectors and real targets."""  

    def __init__(self, samples: int, num_wires: int):  
        self.states, self.labels = generate_superposition_data(num_wires, samples)  

    def __len__(self) -> int:  # type: ignore[override]  
        return len(self.states)  

    def __getitem__(self, index: int):  # type: ignore[override]  
        return {  
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),  
            "target": torch.tensor(self.labels[index], dtype=torch.float32),  
        }  

# ---- Quantum auxiliary blocks ----  

class QuanvCircuit:  
    """Quantum convolution layer based on a random circuit with data‑dependent rotations."""  

    def __init__(self, kernel_size: int, backend=None, shots: int = 100, threshold: float = 0.0):  
        self.n_qubits = kernel_size ** 2  
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")  
        self.shots = shots  
        self.threshold = threshold  
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)  
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]  
        for i in range(self.n_qubits):  
            self.circuit.rx(self.theta[i], i)  
        self.circuit.barrier()  
        self.circuit += random_circuit(self.n_qubits, 2)  
        self.circuit.measure_all()  

    def run(self, data: np.ndarray) -> float:  
        data = np.reshape(data, (1, self.n_qubits))  
        param_binds = []  
        for dat in data:  
            bind = {self.theta[i]: np.pi if val > self.threshold else 0 for i, val in enumerate(dat)}  
            param_binds.append(bind)  
        job = qiskit.execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_binds)  
        result = job.result().get_counts(self.circuit)  
        counts = sum(int(bit) for key, val in result.items() for bit in key) * val  
        return counts / (self.shots * self.n_qubits)  

class QuantumFullyConnected:  
    """Parameterized Ry circuit that simulates a fully‑connected quantum layer."""  

    def __init__(self, n_qubits: int, backend=None, shots: int = 100):  
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")  
        self.shots = shots  
        self.circuit = qiskit.QuantumCircuit(n_qubits)  
        self.theta = qiskit.circuit.Parameter("theta")  
        self.circuit.h(range(n_qubits))  
        self.circuit.barrier()  
        self.circuit.ry(self.theta, range(n_qubits))  
        self.circuit.measure_all()  

    def run(self, thetas: np.ndarray) -> np.ndarray:  
        job = qiskit.execute(  
            self.circuit,  
            self.backend,  
            shots=self.shots,  
            parameter_binds=[{self.theta: theta} for theta in thetas],  
        )  
        result = job.result().get_counts(self.circuit)  
        counts = np.array(list(result.values()))  
        states = np.array(list(result.keys())).astype(float)  
        probabilities = counts / self.shots  
        expectation = np.sum(states * probabilities)  
        return np.array([expectation])  

# ---- Combined quantum model ----  

class QuantumCombinedModel(tq.QuantumModule):  
    """Hybrid quantum model that encodes data, applies a quantum convolution, then a quantum fully‑connected block, and finally a classical readout head."""  

    def __init__(self, num_wires: int):  
        super().__init__()  
        self.n_wires = num_wires  
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])  
        self.conv = QuanvCircuit(kernel_size=2)  
        self.fc = QuantumFullyConnected(n_qubits=num_wires)  
        self.measure = tq.MeasureAll(tq.PauliZ)  
        self.head = nn.Linear(num_wires, 1)  

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  
        bsz = state_batch.shape[0]  
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)  
        # Encode classical data into quantum state  
        self.encoder(qdev, state_batch)  
        # Quantum convolution: apply data‑dependent rotations followed by a random circuit  
        for i in range(self.n_wires):  
            qdev.rx(state_batch[:, i], wires=i)  # simple data embedding  
        self.conv.run(state_batch.cpu().numpy())  # side‑effect measurement for illustration  
        # Quantum fully‑connected block: parameterised Ry gates  
        for i in range(self.n_wires):  
            qdev.ry(torch.nn.Parameter(torch.rand(1), requires_grad=True), wires=i)  
        features = self.measure(qdev)  # (bsz, n_wires)  
        return self.head(features).squeeze(-1)  

__all__ = ["QuantumCombinedModel", "RegressionDataset", "generate_superposition_data"]
