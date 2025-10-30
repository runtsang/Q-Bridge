import numpy as np
import torch
import torch.nn as nn

import qiskit
from qiskit import assemble, transpile
from qiskit.quantum_info import Statevector, Pauli

class QuantumCircuit(nn.Module):
    """Wrapper around a parametrised single‑qubit circuit executed on Aer."""
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(all_qubits)
        self.circuit.barrier()
        self.circuit.ry(self.theta, all_qubits)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probabilities = counts / self.shots
            return np.sum(states * probabilities)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and a quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.quantum_circuit = circuit
        expectation_z = ctx.quantum_circuit.run(inputs.tolist())
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        input_values = np.array(inputs.tolist())
        shift = np.ones_like(input_values) * ctx.shift
        gradients = []
        for idx, value in enumerate(input_values):
            expectation_right = ctx.quantum_circuit.run([value + shift[idx]])
            expectation_left = ctx.quantum_circuit.run([value - shift[idx]])
            gradients.append(expectation_right - expectation_left)
        gradients = torch.tensor([gradients]).float()
        return gradients * grad_output.float(), None, None

class QuantumAutoencoder(nn.Module):
    """Simple quantum autoencoder that maps an image to a latent vector
    using a shallow circuit of RY gates and Z‑expectation values."""
    def __init__(self, n_qubits: int = 32, backend=None) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("statevector_simulator")
        self.base_circuit = qiskit.QuantumCircuit(n_qubits)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (B,3,32,32)
        B = inputs.shape[0]
        # Compute 8×8 patches and average over channels
        patches = inputs.reshape(B, 3, 8, 8, 4, 4).mean(-1).mean(-1).mean(1)  # (B,8,8)
        angles = patches.reshape(B, -1)  # (B,64)
        # Scale to [-π/2, π/2]
        angles = (angles / 255.0) * np.pi - np.pi / 2
        latent_vectors = []
        for theta in angles:
            circuit = self.base_circuit.copy()
            for q, angle in enumerate(theta):
                circuit.ry(angle, q)
            state = Statevector.from_instruction(circuit)
            exps = []
            for q in range(self.n_qubits):
                exp = state.expectation_value(Pauli('Z').to_matrix(), [q]).real
                exps.append(exp)
            latent_vectors.append(exps)
        return torch.tensor(latent_vectors, dtype=torch.float32)

class AutoEncoderHybridNet(nn.Module):
    """Hybrid classifier that first compresses the input with a quantum
    autoencoder and then classifies using a small MLP followed by
    a quantum expectation head."""
    def __init__(self, latent_dim: int = 64, hidden_dims=(32, 16), shift: float = 0.0) -> None:
        super().__init__()
        self.autoencoder = QuantumAutoencoder()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dims[1], 1),
        )
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.quantum_head = QuantumCircuit(1, backend, shots=100)
        self.hybrid = HybridFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = self.autoencoder(inputs)
        logits = self.classifier(latent)
        probs = self.hybrid(logits, self.quantum_head, 0.0)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["AutoEncoderHybridNet", "QuantumAutoencoder", "QuantumCircuit", "HybridFunction"]
