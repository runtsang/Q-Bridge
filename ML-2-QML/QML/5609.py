import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import math
from typing import Iterable, Tuple, List

# ---------- Quantum quanvolution filter ----------
class QuanvolutionFilter(tq.QuantumModule):
    """
    Quantum filter that encodes a 2x2 image patch into a 4‑qubit circuit
    and measures the expectation of Pauli‑Z on all wires.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_wires = kernel_size ** 2
        # Encoder maps each pixel to an Ry rotation
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "ry", "wires": [i]}
                for i in range(self.n_wires)
            ]
        )
        # A random variational layer
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)
        # Reshape to 2x2 patches
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

# ---------- Quantum classifier circuit builder ----------
def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[tq.QuantumCircuit, Iterable, Iterable, List[tq.SparsePauliOp]]:
    """
    Build a variational quantum circuit that mirrors the classical build_classifier_circuit.
    Returns the circuit, parameter lists, and measurement observables.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import SparsePauliOp

    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    # Data encoding
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

# ---------- Hybrid classifier ----------
class QuanvolutionClassifier(nn.Module):
    """
    Quantum‑enhanced classifier that uses the QuanvolutionFilter as a feature extractor
    and a variational quantum circuit as the classification head.
    """
    def __init__(
        self,
        num_qubits: int = 16,
        depth: int = 3,
        use_transformer: bool = False,
        num_classes: int = 10,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_blocks: int = 2,
        ffn_dim: int = 256,
    ) -> None:
        super().__init__()
        self.filter = QuanvolutionFilter()
        if use_transformer:
            # Simple quantum‑augmented transformer head
            seq_len = 14 * 14
            self.proj = nn.Linear(seq_len, embed_dim)
            self.transformer = nn.Sequential(
                *[tq.QuantumModule() for _ in range(num_blocks)]  # placeholder for quantum transformer blocks
            )
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            # Variational quantum classifier
            self.circuit, self.enc_params, self.weight_params, self.observables = build_classifier_circuit(num_qubits, depth)
            self.q_device = tq.QuantumDevice(num_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.filter(x)
        if hasattr(self, "proj"):
            seq = self.proj(features).unsqueeze(1)
            # In a full implementation, seq would feed into quantum transformer blocks
            logits = self.classifier(seq.squeeze(1))
        else:
            # Execute the variational circuit on the feature vector
            qdev = self.q_device.copy(bsz=features.size(0), device=features.device)
            # Bind parameters
            param_binds = []
            for i, val in enumerate(features.squeeze(1).tolist()):
                bind = {self.enc_params[i]: val}
                param_binds.append(bind)
            job = tq.execute(self.circuit, self.q_device, param_binds=param_binds, shots=1024)
            result = job.get_counts()
            probs = []
            for key in result:
                prob = result[key] / 1024
                probs.append(prob)
            logits = torch.tensor(probs).float().to(features.device)
            logits = logits.view(features.size(0), -1)
        return F.log_softmax(logits, dim=-1)

__all__ = [
    "QuanvolutionFilter",
    "QuanvolutionClassifier",
    "build_classifier_circuit",
]
