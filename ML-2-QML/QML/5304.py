"""QuanvolutionHybrid: classical backbone with optional quantum filter and head implemented with Qiskit.

The module mirrors the TorchQuantum implementation but uses a Qiskit simulator for the
quantum blocks.  It supports side‑by‑side training of the classical Conv2d filter
and a parameterised random circuit that processes 2×2 patches.  An optional
quantum head aggregates the patch features, runs a small variational circuit
and maps the resulting probabilities to class logits.

Typical usage::

    from quanvolution_gen190 import QuanvolutionHybrid
    model = QuanvolutionHybrid(use_quantum=True, use_quantum_head=True)
    logits = model(x)

"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import Aer, execute
from qiskit.circuit.random import random_circuit

__all__ = ["QuanvolutionHybrid", "QuanvolutionQuantumFilterQiskit", "QuanvolutionQuantumHeadQiskit"]

class QuanvolutionQuantumFilterQiskit(nn.Module):
    """Quantum filter that processes 2×2 patches with a parameterised random circuit."""
    def __init__(self, backend=None, shots: int = 1024, threshold: float = 0.5):
        super().__init__()
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold
        self.n_qubits = 4

        # Build a single circuit that will be reused for every patch
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def _run_single(self, data: np.ndarray) -> float:
        """Execute the circuit for a single 4‑element patch."""
        bind = {self.theta[i]: np.pi if val > self.threshold else 0 for i, val in enumerate(data)}
        job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=[bind])
        result = job.result()
        counts = result.get_counts(self.circuit)
        total_ones = 0
        for key, val in counts.items():
            total_ones += key.count("1") * val
        return total_ones / (self.shots * self.n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B,1,28,28)
        bsz = x.shape[0]
        device = x.device
        x_np = x.detach().cpu().numpy().squeeze(1)  # (B,28,28)

        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = np.stack(
                    [
                        x_np[:, r, c],
                        x_np[:, r, c + 1],
                        x_np[:, r + 1, c],
                        x_np[:, r + 1, c + 1],
                    ],
                    axis=1,
                )  # (B,4)
                meas = np.array([self._run_single(d) for d in data])
                patches.append(meas)
        result = np.concatenate(patches, axis=1)  # (B,196)
        return torch.tensor(result, device=device, dtype=torch.float32)

class QuanvolutionQuantumHeadQiskit(nn.Module):
    """Quantum head that maps aggregated patch features to logits."""
    def __init__(self, backend=None, shots: int = 1024):
        super().__init__()
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.n_qubits = 4

        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, 1)
        self.circuit.measure_all()

        self.linear = nn.Linear(self.n_qubits, 10)

    def _run_single(self, features: np.ndarray) -> np.ndarray:
        """Run the circuit for a single 4‑element aggregated feature vector."""
        bind = {self.theta[i]: val for i, val in enumerate(features)}
        job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=[bind])
        result = job.result()
        counts = result.get_counts(self.circuit)
        probs = np.zeros(self.n_qubits)
        for key, val in counts.items():
            for i, bit in enumerate(key):
                if bit == "1":
                    probs[i] += val
        probs /= self.shots
        return probs

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features shape: (B, 4*14*14)
        B = features.shape[0]
        # aggregate over patches to obtain 4 values per sample
        agg = features.view(B, -1, 4).mean(dim=1).detach().cpu().numpy()
        probs = np.array([self._run_single(f) for f in agg])  # (B,4)
        probs_tensor = torch.tensor(probs, device=features.device, dtype=torch.float32)
        logits = self.linear(probs_tensor)
        return logits

class QuanvolutionHybrid(nn.Module):
    """Hybrid model that can be run classically or with quantum layers."""
    def __init__(self, use_quantum: bool = True, use_quantum_head: bool = False,
                 backend=None, shots: int = 1024, threshold: float = 0.5):
        super().__init__()
        self.use_quantum = use_quantum
        self.use_quantum_head = use_quantum_head

        self.classical_filter = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        self.quantum_filter = QuanvolutionQuantumFilterQiskit(backend, shots, threshold)

        if use_quantum_head:
            self.head = QuanvolutionQuantumHeadQiskit(backend, shots)
        else:
            self.head = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_quantum:
            features = self.quantum_filter(x)
        else:
            features = self.classical_filter(x).view(x.size(0), -1)
        logits = self.head(features)
        return F.log_softmax(logits, dim=-1)
