import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute

class FCL(nn.Module):
    """
    Hybrid fully‑connected layer that combines a classical linear map, an optional
    convolution filter, and a parameterized quantum circuit.  The quantum part is
    simulated using Qiskit Aer for fast prototyping.
    """
    def __init__(self, n_features: int = 1, n_qubits: int = 1,
                 use_conv: bool = False, clip: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        if use_conv:
            # Simple 2×2 convolution that mimics the QuanvolutionFilter
            self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        self.n_qubits = n_qubits
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = 100
        self.clip = clip

    def _quantum_expectation(self, thetas):
        # Build a random‑layer circuit with parameterised Ry gates
        theta_params = [qiskit.circuit.Parameter(f"theta_{i}") for i in range(self.n_qubits)]
        qc = QuantumCircuit(self.n_qubits)
        qc.h(range(self.n_qubits))
        for i, param in enumerate(theta_params):
            qc.ry(param, i)
        # Random entangling layer (8 CX gates)
        for _ in range(8):
            q, p = np.random.choice(self.n_qubits, 2, replace=False)
            qc.cx(q, p)
        qc.measure_all()
        param_bind = {theta_params[i]: t for i, t in enumerate(thetas)}
        job = execute(qc, self.backend, shots=self.shots,
                      parameter_binds=[param_bind])
        result = job.result()
        counts = result.get_counts()
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()])
        expectation = np.sum(states * probs) / (2**self.n_qubits - 1)
        return expectation

    def forward(self, thetas):
        thetas_tensor = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        lin_out = torch.tanh(self.linear(thetas_tensor)).mean()
        q_exp = self._quantum_expectation(thetas)
        out = lin_out + q_exp
        if self.clip:
            out = torch.clamp(out, -5.0, 5.0)
        return np.array([out.item()])

__all__ = ["FCL"]
