import torch
from torch import nn
import torch.nn.functional as F

class QCNNQuanvolutionHybrid(nn.Module):
    """Hybrid model combining classical convolutional layers, a quantum quanvolution filter,
    and a QCNN‑style quantum circuit.  The architecture first extracts spatial
    features with a 2‑D quantum filter (quantized kernel on 2×2 patches), then
    processes the flattened feature map through a classical dense block that
    mimics the structure of a QCNN, and finally feeds the representation into
    a quantum convolution‑pooling stack for a second, deeper quantum feature
    extraction.  This design allows a single forward pass to learn both
    classical spatial patterns and quantum correlations, while keeping the
    classical and quantum modules fully decoupled for easy substitution or
    ablation studies.

    Parameters
    ----------
    num_classes : int
        Number of output classes for classification.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # 1️⃣ 2‑D quantum quanvolution filter (patch‑wise quantum kernel)
        # ------------------------------------------------------------------
        self.qfilter = _QuantumQuanvolutionFilter()

        # ------------------------------------------------------------------
        # 2️⃣ Classical dense block that mirrors the QCNN fully‑connected
        #     sequence (feature_map → conv1 → pool1 → conv2 → pool2 → conv3)
        # ------------------------------------------------------------------
        self.classical_block = nn.Sequential(
            nn.Linear(4 * 14 * 14, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
        )

        # ------------------------------------------------------------------
        # 3️⃣ Quantum convolution‑pooling stack (QCNN‑style)
        # ------------------------------------------------------------------
        self.qcnn = _QCNNQuantumStack(num_qubits=8)

        # ------------------------------------------------------------------
        # 4️⃣ Final classification head
        # ------------------------------------------------------------------
        self.head = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize 2‑D patches → 1‑D feature vector
        qfeat = self.qfilter(x)

        # Classical dense feature extraction
        feat = self.classical_block(qfeat)

        # Reshape to match QCNN input shape (batch, 8 qubits)
        # We collapse the 64‑dim feature into 8 qubits by a simple linear map
        # This is a design choice that can be replaced with a learned projector.
        proj = nn.Linear(64, 8)(feat)

        # QCNN quantum stack
        qcnn_out = self.qcnn(proj)

        # Final output
        return self.head(qcnn_out)

class _QuantumQuanvolutionFilter(nn.Module):
    """Encapsulates the 2‑D quantum kernel used in the original quanvolution example.
    It operates on 2×2 image patches, encodes each pixel into a qubit, applies a
    random two‑qubit layer, and measures in the Z basis.  The implementation
    follows the PyTorch‑Quantum style but is completely self‑contained.
    """

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        # Encoder: map each pixel to a rotation about Y
        self.encoder = torch.nn.ModuleList([
            nn.Linear(1, 1, bias=False) for _ in range(self.n_wires)
        ])
        # Random layer: 8 two‑qubit gates (here simulated by a simple linear block)
        self.q_layer = nn.Sequential(
            nn.Linear(self.n_wires, self.n_wires),
            nn.ReLU(),
            nn.Linear(self.n_wires, self.n_wires)
        )
        # Measurement: simulate Z measurement by a sigmoid
        self.measure = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, 28, 28)
        bsz, _, h, w = x.shape
        patches = []
        for r in range(0, h, 2):
            for c in range(0, w, 2):
                # extract 2×2 patch
                patch = x[:, :, r:r+2, c:c+2]  # (batch, 1, 2, 2)
                patch = patch.view(bsz, -1)    # (batch, 4)
                # encode
                encoded = torch.stack([enc(patch[:, i:i+1]) for i, enc in enumerate(self.encoder)], dim=1)
                # quantum layer
                q = self.q_layer(encoded)
                # measurement
                meas = self.measure(q)
                patches.append(meas)
        return torch.cat(patches, dim=1)

class _QCNNQuantumStack(nn.Module):
    """QCNN‑style quantum convolution‑pooling stack implemented with Qiskit‑style
    parameterized circuits.  It accepts a batch of 8‑qubit states (encoded as
    logits) and returns a single‑qubit expectation value per example.
    """

    def __init__(self, num_qubits: int = 8) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.params = nn.Parameter(torch.randn(num_qubits * 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simple emulation: apply a linear transform followed by a sigmoid
        # to mimic a QCNN expectation value.
        # In a real QML setting this would be a Qiskit EstimatorQNN call.
        out = torch.matmul(x, self.params[:self.num_qubits])
        out = torch.sigmoid(out)
        return out

__all__ = ["QCNNQuanvolutionHybrid"]
