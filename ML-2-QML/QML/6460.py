import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import torchquantum as tq

class QuantumSelfAttention:
    """Variational self‑attention circuit built with Qiskit."""
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        # Random variational parameters
        self.rotation_params = np.random.rand(3 * n_qubits)
        self.entangle_params = np.random.rand(n_qubits - 1)

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

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """
        Execute the circuit and return expectation values of Pauli‑Z on each qubit.
        The `inputs` array is ignored in this simplified variant but kept for API compatibility.
        """
        circuit = self._build_circuit(self.rotation_params, self.entangle_params)
        job = qiskit.execute(circuit, self.backend, shots=1024)
        counts = job.result().get_counts(circuit)
        total = sum(counts.values())
        expectations = []
        for i in range(self.n_qubits):
            exp = 0.0
            for bitstring, count in counts.items():
                if bitstring[self.n_qubits - 1 - i] == '0':
                    exp += count
                else:
                    exp -= count
            exp /= total
            expectations.append(exp)
        return np.array(expectations, dtype=np.float32)

class QuanvolutionFilter(tq.QuantumModule):
    """Quantum 2×2 patch extractor based on a random two‑qubit kernel."""
    def __init__(self):
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
        return torch.cat(patches, dim=1)  # (batch, 4*14*14)

class QuanvolutionSelfAttentionClassifier(nn.Module):
    """Quantum‑enhanced classifier that applies a quanvolution filter followed by a Qiskit self‑attention layer."""
    def __init__(self):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.attention = QuantumSelfAttention(n_qubits=4)
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantum patch extraction
        features = self.qfilter(x)  # (batch, 4*14*14)
        batch = features.shape[0]
        seq_len = 14 * 14
        embed_dim = 4
        features = features.view(batch, seq_len, embed_dim)

        # Global mean embedding per sample (used as input to the attention circuit)
        mean_embeds = features.mean(dim=1).cpu().detach().numpy()  # (batch, embed_dim)

        # Run the quantum attention circuit for each sample
        attn_weights = []
        for embed in mean_embeds:
            weights = self.attention.run(embed)
            attn_weights.append(weights)
        attn_weights = torch.tensor(attn_weights, device=features.device, dtype=torch.float32)

        # Modulate patch embeddings with the attention weights
        features = features * attn_weights.unsqueeze(1)
        features = features.view(batch, -1)

        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionSelfAttentionClassifier"]
