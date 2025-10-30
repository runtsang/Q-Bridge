import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid that maps quantum expectations to probabilities."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None


class QuantumNATGen167(nn.Module):
    """Hybrid CNN‑QNN for binary classification/regression.

    Architecture
    ------------
    * Two convolutional layers with ReLU and max‑pooling.
    * A fully‑connected projection to 16×7×7 features.
    * A 4‑qubit quantum block (RandomLayer + trainable RX/RZ/CRX, Hadamard–SX–CNOT).
    * Batch‑norm on the quantum output.
    * Differentiable sigmoid head (optional) to produce class probabilities.
    """
    class HybridQuantumBlock(tq.QuantumModule):
        """Quantum module that encodes classical features and returns expectation values."""
        def __init__(self, n_wires: int = 4):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)
            self.measure = tq.MeasureAll(tq.PauliZ)
            self.norm = nn.BatchNorm1d(self.n_wires)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

        def forward_features(self, features: torch.Tensor) -> torch.Tensor:
            bsz = features.shape[0]
            qdev = tq.QuantumDevice(
                n_wires=self.n_wires,
                bsz=bsz,
                device=features.device,
                record_op=True,
            )
            self.encoder(qdev, features)
            self.forward(qdev)
            out = self.measure(qdev)
            return self.norm(out)

    def __init__(self, sigmoid_shift: float = 0.0, use_sigmoid: bool = True):
        super().__init__()
        self.sigmoid_shift = sigmoid_shift
        self.use_sigmoid = use_sigmoid

        # Classical backbone
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc_proj = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.bn = nn.BatchNorm1d(4)

        # Quantum head
        self.quantum_block = self.HybridQuantumBlock(n_wires=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning class probabilities or regression logits."""
        # Classical feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_proj(x)
        x = self.bn(x)

        # Quantum expectation
        q_out = self.quantum_block.forward_features(x)

        # Optional sigmoid head
        if self.use_sigmoid:
            probs = HybridFunction.apply(q_out, self.sigmoid_shift)
            return torch.cat((probs, 1 - probs), dim=-1)
        else:
            return q_out

__all__ = ["QuantumNATGen167"]
