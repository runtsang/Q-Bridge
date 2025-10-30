import torch
import torch.nn as nn
import numpy as np
import torchquantum as tq
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class QuanvolutionFilter(tq.QuantumModule):
    """Quantum 2×2 kernel that maps image patches onto a 4‑qubit register."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class QuantumAutoencoder(tq.QuantumModule):
    """Quantum auto‑encoder that compresses a feature vector into a 2‑dimensional latent."""
    def __init__(self) -> None:
        super().__init__()
        self.weight_params = ParameterVector("weight", 2)
        qc = QuantumCircuit(2)
        qc.ry(self.weight_params[0], 0)
        qc.ry(self.weight_params[1], 1)
        qc.cx(0, 1)
        self.sampler = Sampler()
        self.qnn = SamplerQNN(
            circuit=qc,
            input_params=[],
            weight_params=self.weight_params,
            sampler=self.sampler,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qnn(x)

class QuantumSelfAttention:
    """Variational self‑attention block implemented with a small qasm circuit."""
    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = Aer.get_backend("qasm_simulator")

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

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 256):
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, self.backend, shots=shots)
        return job.result().get_counts(circuit)

class HybridQuanvolutionNet(tq.QuantumModule):
    """Hybrid quantum‑classical network that chains quanvolution, quantum auto‑encoding,
    self‑attention, and a final classical classifier."""
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.autoencoder = QuantumAutoencoder()
        self.attention = QuantumSelfAttention(n_qubits=4)
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Quantum quanvolution
        features = self.qfilter(x)                      # (B, 4*14*14)
        # 2. Quantum auto‑encoder (latent extraction)
        latent = self.autoencoder(features[:, :2])      # (B, 2)
        # 3. Self‑attention on the latent vector
        rotation_params = np.tile(latent.cpu().numpy(), (1, 6))  # 4 qubits * 3 params each
        entangle_params = np.random.randn(features.shape[0], 3)   # random entangle params per sample
        attn_counts = [self.attention.run(rotation_params[i], entangle_params[i]) for i in range(features.shape[0])]
        # Convert counts to a probability vector per batch
        probs = []
        for counts in attn_counts:
            total = sum(counts.values())
            probs.append([counts.get(f'{i:04b}', 0) / total for i in range(16)])
        attn_tensor = torch.as_tensor(probs, dtype=torch.float32, device=x.device)
        # 4. Final classification
        logits = self.classifier(attn_tensor)
        return logits

__all__ = ["HybridQuanvolutionNet"]
