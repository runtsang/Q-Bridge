import numpy as np
import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit_aer import AerSimulator

class QuantumSelfAttention:
    """Quantum circuit implementing a self‑attention style block."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = AerSimulator()

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024) -> np.ndarray:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = self.backend.run(circuit, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)
        # Convert measurement results to a float vector (0/1 counts)
        vec = np.zeros(self.n_qubits, dtype=np.float32)
        for bitstring, count in counts.items():
            for idx, bit in enumerate(reversed(bitstring)):
                vec[idx] += count * int(bit)
        vec /= shots
        return vec

class QuantumRefinement:
    """Variational circuit that refines a classical latent vector."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.backend = AerSimulator(method='statevector')

    def refine(self, latent: np.ndarray) -> np.ndarray:
        # Map latent values to rotation angles in [-π, π]
        angles = 2 * np.pi * (latent - 0.5)
        circuit = QuantumCircuit(self.n_qubits)
        for i, angle in enumerate(angles):
            circuit.rx(angle, i)
        result = self.backend.run(circuit).result()
        state = result.get_statevector(circuit)
        # Compute expectation value of Pauli‑Z on each qubit
        expz = []
        for i in range(self.n_qubits):
            exp = 0.0
            for idx, amp in enumerate(state):
                if ((idx >> i) & 1) == 1:
                    exp -= abs(amp) ** 2
                else:
                    exp += abs(amp) ** 2
            expz.append(exp)
        expz = np.array(expz, dtype=np.float32)
        # Transform to [0, 1] range
        refined = (expz + 1) / 2
        return refined

class UnifiedAutoencoder(nn.Module):
    """Hybrid quantum‑classical autoencoder."""
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        # Classical encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], latent_dim),
        )
        # Quantum refinement blocks
        self.quantum_attention = QuantumSelfAttention(n_qubits=latent_dim)
        self.quantum_refinement = QuantumRefinement(n_qubits=latent_dim)
        # Classical decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical encoding
        latent = self.encoder(x).detach().cpu().numpy()
        # Quantum self‑attention refinement
        attn_params = np.random.rand(3 * self.latent_dim)  # rotation params
        ent_params = np.random.rand(self.latent_dim - 1)   # entanglement params
        attn_out = self.quantum_attention.run(attn_params, ent_params)
        # Combine classical latent with quantum attention output
        combined = latent + attn_out  # broadcast across batch
        # Quantum refinement for each sample
        refined = np.array([self.quantum_refinement.refine(sample) for sample in combined])
        # Convert back to torch tensor
        refined_tensor = torch.from_numpy(refined).to(x.device)
        return self.decoder(refined_tensor)
