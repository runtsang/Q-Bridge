from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from qiskit import Aer, transpile, assemble, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes

# ----- Quantum Autoencoder Circuit -----
class QuantumAutoencoderCircuit:
    """
    Parameterized two‑qubit autoencoder with swap‑test for feature compression.
    """
    def __init__(self,
                 num_latent: int,
                 num_trash: int,
                 shots: int = 100,
                 backend=None) -> None:
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.shots = shots
        self.backend = backend or Aer.get_backend("aer_simulator")
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        total_qubits = self.num_latent + 2 * self.num_trash + 1
        qc = QuantumCircuit(total_qubits, 1)
        num_params = self.num_latent + self.num_trash

        # Ry gates parametrised by the classical input
        for i in range(num_params):
            qc.ry(Parameter(f"theta_{i}"), i)

        # Ansatz
        ansatz = RealAmplitudes(num_params, reps=5)
        qc.compose(ansatz, range(num_params), inplace=True)
        qc.barrier()

        aux = self.num_latent + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        qc.h(aux)

        qc.measure(aux, 0)
        return qc

    def run(self, params: np.ndarray) -> float:
        bind_dict = {f"theta_{i}": float(params[i]) for i in range(len(params))}
        bound_qc = self.circuit.bind_parameters(bind_dict)
        compiled = transpile(bound_qc, self.backend)
        qobj = assemble(compiled, shots=self.shots)
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        return counts.get("1", 0) / self.shots

# ----- Differentiable Quantum Layer -----
class HybridQuantumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumAutoencoderCircuit):
        probs = []
        for sample in inputs.cpu().numpy():
            probs.append(circuit.run(sample))
        probs = torch.tensor(probs, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs)
        ctx.circuit = circuit
        return probs.unsqueeze(-1)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, = ctx.saved_tensors
        circuit = ctx.circuit
        shift = np.pi / 2
        grads = []
        for sample in inputs.cpu().numpy():
            grad_sample = []
            for i, _ in enumerate(sample):
                right = sample.copy()
                left = sample.copy()
                right[i] += shift
                left[i] -= shift
                grad_sample.append((circuit.run(right) - circuit.run(left)) / 2.0)
            grads.append(grad_sample)
        grads = torch.tensor(grads, dtype=torch.float32, device=inputs.device)
        return grads * grad_output.squeeze(-1).unsqueeze(-1), None

# ----- Hybrid Classifier -----
class HybridAutoencoderClassifier(nn.Module):
    """
    Quantum hybrid classifier: classical CNN + parameterised quantum autoencoder.
    The CNN extracts features; the quantum circuit compresses them and the auxiliary qubit expectation yields the binary probability.
    """
    def __init__(self,
                 image_shape: Tuple[int, int, int],
                 num_latent: int,
                 num_trash: int,
                 backend=None) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(image_shape[0], 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(0.5),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *image_shape)
            cnn_out = self.cnn(dummy)
        cnn_feat_dim = cnn_out.shape[1]

        self.fc = nn.Linear(cnn_feat_dim, num_latent + num_trash)
        self.quantum_circuit = QuantumAutoencoderCircuit(num_latent, num_trash, shots=200, backend=backend)
        self.shift = np.pi / 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        cnn_feat = self.cnn(x)
        params = torch.relu(self.fc(cnn_feat))
        probs = HybridQuantumFunction.apply(params, self.quantum_circuit)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["HybridAutoencoderClassifier"]
