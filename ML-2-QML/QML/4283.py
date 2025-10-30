import pennylane as qml
import torch
import torch.nn as nn

# Quantum device with 4 qubits
dev = qml.device("default.qubit", wires=4)

def _encode_patch(patch: torch.Tensor):
    """Encode a 2×2 patch into 4 qubits using Ry rotations."""
    for i, val in enumerate(patch.flatten()):
        qml.RY(val, wires=i)

@qml.qnode(dev, interface="torch")
def quanvolution_circuit(patch: torch.Tensor):
    """Processes a single 2×2 patch and returns a 4‑dimensional feature vector."""
    _encode_patch(patch)
    # Simple entanglement
    qml.CX(wires=[0, 1])
    qml.CX(wires=[2, 3])
    return torch.stack([qml.expval(qml.PauliZ(i)) for i in range(4)])

@qml.qnode(dev, interface="torch")
def fully_connected_circuit(features: torch.Tensor):
    """Aggregates the first 4 patch features into a single expectation value."""
    _encode_patch(features)
    qml.CX(wires=[0, 1])
    qml.CX(wires=[2, 3])
    return torch.mean([qml.expval(qml.PauliZ(i)) for i in range(4)])

@qml.qnode(dev, interface="torch")
def sampler_circuit(inputs: torch.Tensor, weights: torch.Tensor):
    """Quantum sampler producing a 2‑dimensional probability distribution."""
    # Encode inputs
    for i in range(2):
        qml.RY(inputs[i], wires=i)
    # Variational layer
    for i in range(4):
        qml.RY(weights[i], wires=i % 2)
    # Entangle
    qml.CX(wires=[0, 1])
    qml.CX(wires=[2, 3])
    probs = torch.stack([qml.expval(qml.PauliZ(i)) for i in range(2)])
    return torch.softmax(probs, dim=-1)

class HybridQuantumNetQML(nn.Module):
    """
    Quantum implementation of the hybrid architecture.
    The network processes 28×28 images through a quantum quanvolution,
    a quantum fully‑connected layer, and a quantum sampler, then
    projects the concatenated representation onto the class space
    with a classical linear head.
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.num_classes = num_classes
        # Trainable linear head
        self.linear_weight = nn.Parameter(torch.randn(4 * 14 * 14 + 1 + 2, num_classes))
        self.linear_bias = nn.Parameter(torch.randn(num_classes))
        # Sampler weights
        self.sampler_weights = nn.Parameter(torch.randn(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Batch of grayscale images of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (batch, num_classes).
        """
        batch = x.shape[0]
        assert x.shape[2:] == (28, 28), "input must be 28×28"

        # 1. Quanvolution stage – process 2×2 patches
        patch_feats = []
        for i in range(0, 28, 2):
            for j in range(0, 28, 2):
                patch = x[:, 0, i:i+2, j:j+2]          # (batch, 2, 2)
                patch_flat = patch.view(batch, -1)      # (batch, 4)
                # Run the quantum circuit for each batch element
                feat_batch = torch.stack([quanvolution_circuit(p) for p in patch_flat], dim=0)
                patch_feats.append(feat_batch)
        # Concatenate all patch features -> shape (batch, 4*14*14)
        qfeat = torch.cat(patch_feats, dim=-1)

        # 2. Quantum fully‑connected layer (classical surrogate)
        qfc_out = torch.stack([fully_connected_circuit(qfeat[i, :4]) for i in range(batch)], dim=0)

        # 3. Quantum sampler layer
        sampler_in = qfeat[:, :2]
        sampler_out = torch.stack([sampler_circuit(sampler_in[i], self.sampler_weights) for i in range(batch)], dim=0)

        # 4. Concatenate all representations
        combined = torch.cat([qfeat, qfc_out.unsqueeze(-1), sampler_out], dim=-1)

        # 5. Classical linear head
        logits = torch.matmul(combined, self.linear_weight) + self.linear_bias
        return torch.log_softmax(logits, dim=-1)

__all__ = ["HybridQuantumNetQML"]
