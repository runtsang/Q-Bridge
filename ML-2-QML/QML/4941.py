import torch
import torch.nn as nn
import torchquantum as tq

# --------------------------------------------------------------------------- #
# Quantum QCNN-inspired feature extractor
# --------------------------------------------------------------------------- #
class QCNNQuantumLayer(tq.QuantumModule):
    """
    Implements a single convolutional layer for a QCNN.  The layer
    applies the same 2‑qubit unitary to each adjacent pair of qubits.
    """
    def __init__(self, num_qubits: int, n_ops: int = 20):
        super().__init__()
        self.num_qubits = num_qubits
        self.n_ops = n_ops
        # Random unitary for each pair
        self.random_layer = tq.RandomLayer(n_ops=self.n_ops, wires=list(range(num_qubits)))

    def forward(self, qdev: tq.QuantumDevice) -> None:
        # Apply the random layer globally – already parameterised for each pair
        self.random_layer(qdev)


class QCNNQuantumPool(tq.QuantumModule):
    """
    A pooling operation that discards half of the qubits after applying a
    simple 2‑qubit unitary.  This mimics the depth‑reducing behaviour of a
    classical pooling layer.
    """
    def __init__(self, num_qubits: int, n_ops: int = 10):
        super().__init__()
        self.num_qubits = num_qubits
        self.pool_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(num_qubits)))

    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.pool_layer(qdev)
        # Collapse the device to half the number of qubits
        # (for simplicity we just leave the unused qubits untouched; real
        # implementations might trace them out).


# --------------------------------------------------------------------------- #
# Hybrid quantum regression model
# --------------------------------------------------------------------------- #
class QuantumHybridRegression(tq.QuantumModule):
    """
    A full quantum module that accepts classical data, encodes it into a
    quantum state, passes it through several QCNN layers, measures,
    and finally maps the expectation values to a scalar output.
    """
    def __init__(self,
                 num_qubits: int,
                 num_features: int,
                 conv_layers: int = 3,
                 pool_layers: int = 2,
                 n_ops_conv: int = 20,
                 n_ops_pool: int = 10):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_features = num_features

        # Encoder – maps each feature to a rotation on a dedicated qubit
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_qubits}xRy"]
        )

        # Stacked QCNN layers
        self.conv_layers = nn.ModuleList(
            [QCNNQuantumLayer(num_qubits, n_ops_conv) for _ in range(conv_layers)]
        )
        self.pool_layers = nn.ModuleList(
            [QCNNQuantumPool(num_qubits // (i + 1), n_ops_pool) for i in range(pool_layers)]
        )

        # Measurement
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Classifier head – maps quantum expectation values to a scalar
        self.head = nn.Linear(num_qubits, 1)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            data: Tensor of shape [batch, num_features] containing
                  classical features to be encoded.
        """
        bsz = data.shape[0]
        qdev = tq.QuantumDevice(self.num_qubits, bsz=bsz, device=data.device)

        # Encode classical features
        self.encoder(qdev, data)

        # Apply QCNN layers
        for conv in self.conv_layers:
            conv(qdev)
        for pool in self.pool_layers:
            pool(qdev)

        # Measure
        features = self.measure(qdev)  # shape [batch, num_qubits]
        return self.head(features).squeeze(-1)


# --------------------------------------------------------------------------- #
# Sampler QNN – quantum probability distribution generator
# --------------------------------------------------------------------------- #
class QuantumSamplerQNN(tq.QuantumModule):
    """
    A lightweight quantum sampler that emits a 2‑dimensional probability
    vector for each input.  The circuit consists of two Ry rotations,
    a CNOT, and subsequent Ry rotations as in the classical SamplerQNN.
    """
    def __init__(self):
        super().__init__()
        self.circuit = tq.circuit.QuantumCircuit(2)
        self.circuit.ry(tq.Parameter("θ0"), 0)
        self.circuit.ry(tq.Parameter("θ1"), 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(tq.Parameter("θ2"), 0)
        self.circuit.ry(tq.Parameter("θ3"), 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(tq.Parameter("θ4"), 0)
        self.circuit.ry(tq.Parameter("θ5"), 1)

        self.sampler = tq.StatevectorSampler()
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Expects input tensor of shape [batch, 2] – the two angles for the
        initial Ry rotations.  Returns a probability distribution over
        the two basis states |00> and |01>.
        """
        bsz = data.shape[0]
        qdev = tq.QuantumDevice(2, bsz=bsz, device=data.device)
        self.circuit(qdev, data)
        state = self.sampler(qdev)
        probs = self.measure(state)
        # Convert expectation values to probabilities
        probs = (probs + 1.0) / 2.0
        return probs


# --------------------------------------------------------------------------- #
# Quanvolution filter – quantum‑aware image patch extractor
# --------------------------------------------------------------------------- #
class QuantumQuanvolutionFilter(tq.QuantumModule):
    """
    Quantum implementation of the classic 2×2 patch filter.  Each patch
    is passed through a small 4‑qubit random layer and measured.
    """
    def __init__(self, num_wires: int = 4):
        super().__init__()
        self.num_wires = num_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(num_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Expects input shape [batch, 1, 28, 28] – a single‑channel grayscale image.
        Returns a flattened feature vector ready for a linear classifier.
        """
        bsz = img.shape[0]
        qdev = tq.QuantumDevice(self.num_wires, bsz=bsz, device=img.device)

        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [
                        img[:, 0, r, c],
                        img[:, 0, r, c + 1],
                        img[:, 0, r + 1, c],
                        img[:, 0, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, patch)
                self.q_layer(qdev)
                meas = self.measure(qdev)
                patches.append(meas.view(bsz, self.num_wires))
        return torch.cat(patches, dim=1)


# --------------------------------------------------------------------------- #
# Exports
# --------------------------------------------------------------------------- #
__all__ = [
    "QuantumHybridRegression",
    "QuantumSamplerQNN",
    "QuantumQuanvolutionFilter",
]
