import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import transpile, assemble

class QuantumKernelCircuit:
    """Parameterised one‑qubit circuit used as a quantum kernel."""
    def __init__(self, backend: qiskit.providers.Provider, shots: int = 1024):
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(1)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(0)
        self.circuit.barrier()
        self.circuit.ry(self.theta, 0)
        self.circuit.measure_all()

    def run(self, theta: float) -> float:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled,
                        shots=self.shots,
                        parameter_binds=[{self.theta: theta}])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        counts = np.array(list(result.values()))
        probs = counts / self.shots
        # expectation of measurement of the first qubit
        return np.sum((np.array(list(result.keys()), dtype=int) % 2) * probs * -1)

class QuantumSelfAttention:
    """Quantum circuit implementing a simple self‑attention style block."""
    def __init__(self, n_qubits: int = 4, backend: qiskit.providers.Provider = None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.qr = qiskit.QuantumRegister(n_qubits, "q")
        self.cr = qiskit.ClassicalRegister(n_qubits, "c")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> qiskit.QuantumCircuit:
        circ = qiskit.QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circ.rx(rotation_params[3 * i], i)
            circ.ry(rotation_params[3 * i + 1], i)
            circ.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circ.crx(entangle_params[i], i, i + 1)
        circ.measure(self.qr, self.cr)
        return circ

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = None) -> np.ndarray:
        circ = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circ, self.backend, shots=shots or self.shots)
        result = job.result().get_counts(circ)
        # expectation values of all qubits
        expectations = []
        for i in range(self.n_qubits):
            key_0 = '0' * i + '0' + '0' * (self.n_qubits - i - 1)
            key_1 = '0' * i + '1' + '0' * (self.n_qubits - i - 1)
            p0 = result.get(key_0, 0) / self.shots
            p1 = result.get(key_1, 0) / self.shots
            expectations.append(p0 * 1 + p1 * -1)
        return np.array(expectations)

class QuantumHybridHead:
    """One‑qubit quantum head producing a scalar expectation."""
    def __init__(self, backend: qiskit.providers.Provider = None, shots: int = 1024):
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(1)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(0)
        self.circuit.rx(self.theta, 0)
        self.circuit.measure_all()

    def run(self, theta: float) -> float:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled,
                        shots=self.shots,
                        parameter_binds=[{self.theta: theta}])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        counts = np.array(list(result.values()))
        probs = counts / self.shots
        return np.sum((np.array(list(result.keys()), dtype=int) % 2) * probs * -1)

class HybridAttentionKernelNet(nn.Module):
    """
    Quantum‑classical hybrid network combining a CNN backbone, a prototype‑based quantum kernel,
    a quantum self‑attention block, and a quantum hybrid head.
    """
    def __init__(self,
                 n_prototypes: int = 10,
                 prototype_dim: int = 84,
                 gamma: float = 1.0,
                 n_qubits_attention: int = 4,
                 shots: int = 1024) -> None:
        super().__init__()
        # Convolutional backbone identical to the classical version
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, prototype_dim)

        # Prototype bank – learnable
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, prototype_dim))

        # Quantum components
        self.kernel_circuit = QuantumKernelCircuit(backend=qiskit.Aer.get_backend("aer_simulator"),
                                                   shots=shots)
        self.attention_block = QuantumSelfAttention(n_qubits=n_qubits_attention,
                                                    backend=qiskit.Aer.get_backend("aer_simulator"),
                                                    shots=shots)
        self.hybrid_head = QuantumHybridHead(backend=qiskit.Aer.get_backend("aer_simulator"),
                                             shots=shots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)

        # Dense layers
        x = F.relu(self.fc1(x))
        feat = self.fc2(x)  # (batch, embed_dim)

        # Quantum self‑attention: use first n_qubits*3 elements of feat as rotation params
        rot = feat[:, :self.attention_block.n_qubits * 3].detach().cpu().numpy()
        ent = np.zeros((rot.shape[0], self.attention_block.n_qubits - 1))
        att_weights = []
        for r, e in zip(rot, ent):
            att_weights.append(self.attention_block.run(r, e))
        att_weights = np.stack(att_weights, axis=0)  # (batch, n_qubits)

        # Quantum kernel similarities between feature and each prototype
        kernel_sims = []
        for p in self.prototypes:
            theta = torch.dot(feat.squeeze(), p).item()
            kernel_sims.append(self.kernel_circuit.run(theta))
        kernel_sims = np.array(kernel_sims)  # (n_prototypes,)

        # Combine attention and kernel signals
        combined = torch.softmax(torch.tensor(kernel_sims, device=feat.device) +
                                 att_weights.sum(dim=1, keepdim=True),
                                 dim=1)  # (batch, n_prototypes)

        # Weighted prototype aggregation
        aggregated = torch.sum(combined.unsqueeze(-1) * self.prototypes, dim=1)  # (batch, embed_dim)

        # Quantum hybrid head: use mean of aggregated vector as parameter
        theta = aggregated.mean(dim=1).detach().cpu().numpy()
        logits = []
        for t in theta:
            logits.append(self.hybrid_head.run(t))
        logits = torch.tensor(logits, device=feat.device)

        probs = torch.sigmoid(logits)
        return torch.stack([probs, 1 - probs], dim=-1)

__all__ = ["HybridAttentionKernelNet"]
