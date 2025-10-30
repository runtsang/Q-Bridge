import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit, assemble, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.circuit import Parameter

class QuantumExpectationHead(nn.Module):
    """Quantum circuit that returns a single expectation value for each input.

    The circuit is a two‑qubit circuit with parameters taken from the input
    vector.  The expectation of Pauli‑Z on the first qubit is returned
    and mapped to a probability via (exp + 1)/2.
    """
    def __init__(self, n_qubits: int = 2, shots: int = 1024, shift: float = np.pi / 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.shots = shots
        self.shift = shift
        self.backend = AerSimulator()
        self.circuit = QuantumCircuit(n_qubits)
        for q in range(n_qubits):
            self.circuit.h(q)
        self.circuit.barrier()
        # parameter for each qubit
        self.params = [Parameter(f'theta_{q}') for q in range(n_qubits)]
        for q, param in enumerate(self.params):
            self.circuit.ry(param, q)
        self.circuit.measure_all()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, n_qubits)
        batch = x.shape[0]
        expectations = []
        for i in range(batch):
            theta_vals = x[i].detach().cpu().numpy()
            param_binds = [{p: v + self.shift for p, v in zip(self.params, theta_vals)}]
            circ = transpile(self.circuit, self.backend)
            qobj = assemble(circ, shots=self.shots, parameter_binds=param_binds)
            job = self.backend.run(qobj)
            result = job.result().get_counts()
            # compute expectation of Z on qubit 0
            exp = 0.0
            for bitstring, count in result.items():
                # bitstring is in little‑endian order
                z = 1 if bitstring[0] == '0' else -1
                exp += z * count / self.shots
            expectations.append(exp)
        expectations = torch.tensor(expectations, dtype=torch.float32, device=x.device)
        # map to probability
        probs = (expectations + 1) / 2
        return probs

class HybridBinaryClassifier(nn.Module):
    """Hybrid quantum‑classical binary classifier.

    The network consists of a classical dense head followed by a quantum
    expectation head.  The two predictions are fused with a learnable
    gate that can be trained to emphasize either branch.

    The forward method returns a probability vector of shape (batch, 2).
    """
    def __init__(self, in_features: int, n_qubits: int = 2, shots: int = 1024, gate_init: float = 0.5):
        super().__init__()
        self.classical_head = nn.Linear(in_features, 2)
        self.quantum_head = QuantumExpectationHead(n_qubits=n_qubits, shots=shots)
        self.gate = nn.Parameter(torch.tensor(gate_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        class_logits = self.classical_head(x)
        quantum_probs = self.quantum_head(x).unsqueeze(-1)  # shape (batch, 1)
        # convert quantum probability to logits: logit = log(p/(1-p))
        quantum_logits = torch.log(quantum_probs / (1 - quantum_probs + 1e-6) + 1e-6)
        gate = torch.sigmoid(self.gate)
        logits = gate * class_logits + (1 - gate) * quantum_logits
        probs = F.softmax(logits, dim=-1)
        return probs

__all__ = ["HybridBinaryClassifier"]
