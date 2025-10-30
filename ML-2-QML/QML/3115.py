import torch
import torch.nn as nn
import pennylane as qml

class HybridConvNet(nn.Module):
    """
    Quantum‑augmented convolutional network that replaces the
    classical quantum‑inspired filter with a variational circuit
    implemented in Pennylane.  The network uses a small
    convolutional backbone, encodes the flattened features
    into a 4‑qubit device, runs a parameter‑shuffled circuit,
    and measures Pauli‑Z expectation values as the final
    four‑dimensional output.
    """
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 4,
                 conv_filters: int = 8,
                 kernel_size: int = 3,
                 num_qubits: int = 4,
                 device: str | None = None):
        super().__init__()
        self.device_str = device or "default.qubit"
        # Classical convolutional backbone
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, conv_filters, kernel_size=kernel_size,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(conv_filters, conv_filters * 2, kernel_size=kernel_size,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(conv_filters * 2, conv_filters * 4, kernel_size=kernel_size,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # Quantum device
        self.num_qubits = num_qubits
        self.dev = qml.device(self.device_str, wires=self.num_qubits)
        # Trainable variational parameters
        self.var_params = nn.Parameter(torch.rand(2 * self.num_qubits))
        # Quantum node
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def qnode(inputs: torch.Tensor, params: torch.Tensor):
            # Angle‑encoding of the flattened features
            for i, wire in enumerate(range(self.num_qubits)):
                qml.RY(inputs[i], wires=wire)
            # Variational block
            for i, wire in enumerate(range(self.num_qubits)):
                qml.RZ(params[i], wires=wire)
                qml.CNOT(wires=[wire, (wire + 1) % self.num_qubits])
            # Measurement
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]
        self.qnode = qnode
        # Fully‑connected head
        conv_out_dim = conv_filters * 4 * 3 * 3  # assume 28×28 input
        self.fc1 = nn.Linear(conv_out_dim + self.num_qubits, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.batch_norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, H, W).

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch, num_classes).
        """
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        # Use first `num_qubits` entries for quantum encoding
        q_inputs = flat[:, :self.num_qubits]
        if q_inputs.shape[1] < self.num_qubits:
            pad = self.num_qubits - q_inputs.shape[1]
            q_inputs = torch.nn.functional.pad(q_inputs, (0, pad))
        # Quantum forward
        q_out = self.qnode(q_inputs, self.var_params)
        q_out = torch.stack(q_out, dim=1)
        # Concatenate with classical features
        concat = torch.cat([flat, q_out], dim=1)
        out = self.fc1(concat)
        out = torch.relu(out)
        out = self.fc2(out)
        return self.batch_norm(out)

__all__ = ["HybridConvNet"]
