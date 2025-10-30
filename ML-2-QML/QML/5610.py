import torch
import torch.nn as nn
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import RandomCircuit

class HybridClassifier(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_qubits: int,
                 num_classes: int,
                 depth: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.num_qubits = num_qubits
        self.num_classes = num_classes
        self.depth = depth
        # Random unitary layer
        self.random_circuit = RandomCircuit(num_qubits, depth=1, seed=42)
        # Variational parameters
        self.var_params = nn.Parameter(torch.randn(num_qubits * depth))
        # Classical classifier head
        self.classifier = nn.Linear(num_qubits, num_classes)
        # Quantum backend
        self.backend = Aer.get_backend('qasm_simulator')
        self.shots = 1024

    def _build_circuit(self, x_vals: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        # Encode classical features into RX rotations
        for i, val in enumerate(x_vals):
            qc.rx(val, i)
        # Random unitary layer
        qc.append(self.random_circuit, range(self.num_qubits))
        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for q in range(self.num_qubits):
                qc.ry(self.var_params[idx].item(), q)
                idx += 1
            for q in range(self.num_qubits - 1):
                qc.cz(q, q + 1)
        qc.measure_all()
        return qc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        logits_batch = []
        for i in range(batch):
            x_vals = x[i].cpu().numpy()
            qc = self._build_circuit(x_vals)
            job = execute(qc, self.backend, shots=self.shots)
            result = job.result()
            counts = result.get_counts()
            exp_vals = []
            for q in range(self.num_qubits):
                exp = 0.0
                for bitstring, count in counts.items():
                    bit = int(bitstring[::-1][q])  # Qiskit bit order
                    exp += ((-1) ** bit) * count
                exp /= self.shots
                exp_vals.append(exp)
            exp_tensor = torch.tensor(exp_vals, dtype=torch.float32, device=x.device)
            logits = self.classifier(exp_tensor)
            logits_batch.append(logits)
        return torch.stack(logits_batch)
