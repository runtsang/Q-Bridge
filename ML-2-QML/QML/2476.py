import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class HybridQuanvolution(tq.QuantumModule):
    """
    Quantum implementation of the hybrid quanvolution filter.
    For each 2×2 patch a random circuit is applied and the average
    probability of measuring |1> across the qubits is computed.
    A threshold turns this into a binary feature, and a linear
    classifier maps the concatenated patch features to logits.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.5,
                 shots: int = 100, backend: str = "qasm_simulator"):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.backend = backend
        self.n_qubits = kernel_size ** 2

        # Random circuit generator
        self.circuit = tq.QuantumCircuit(self.n_qubits)
        self.theta = [tq.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += tq.random_circuit(self.n_qubits, depth=2)
        self.circuit.measure_all()

        # Encoder that maps classical data to Ry rotations
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_qubits)]
        )
        # Measurement in Z basis
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Linear head
        self.linear = nn.Linear(self.n_qubits * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of 28×28 grayscale images.
        """
        B, C, H, W = x.shape
        device = x.device
        # Quantum device with batch support
        qdev = tq.QuantumDevice(self.n_qubits, bsz=B, device=device)

        # Reshape input into 2×2 patches
        x = x.view(B, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, r:r + 2, c:c + 2]  # (B, 2, 2)
                data = patch.reshape(B, self.n_qubits)  # (B, 4)

                # Encode classical data into qubits via Ry rotations
                self.encoder(qdev, data)
                # Apply the random circuit
                self.circuit(qdev)
                # Measure all qubits in Z basis
                meas = self.measure(qdev)  # (B, n_qubits)
                # Convert expectation values to probabilities of |1>
                probs = (1 - meas) / 2
                # Average probability across qubits
                avg = probs.mean(dim=1)  # (B,)
                # Threshold to binary feature
                feat = torch.where(avg > self.threshold,
                                   torch.ones_like(avg),
                                   torch.zeros_like(avg))
                patches.append(feat)

        # Concatenate all patch features
        feat_vec = torch.stack(patches, dim=1).view(B, -1)
        logits = self.linear(feat_vec)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQuanvolution"]
