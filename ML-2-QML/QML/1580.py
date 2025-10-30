import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class HybridQuanvolution(nn.Module):
    """Hybrid classical‑quantum filter and classifier with a parameterised QAOA‑style circuit."""
    def __init__(self) -> None:
        super().__init__()
        # Quantum device
        self.dev = qml.device("default.qubit", wires=4)
        # Trainable parameters for the quantum layer
        self.params = nn.Parameter(torch.randn(4))
        # Linear head to produce class logits
        self.linear = nn.Linear(4 * 14 * 14, 10)

        @qml.qnode(self.dev, interface="torch")
        def circuit(patch: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            # Encode the patch into rotation angles
            for i in range(4):
                qml.RY(patch[:, i], wires=i)
            # Parameterised rotation layer
            for i in range(4):
                qml.RY(params[i], wires=i)
            # Entangling layer
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[2, 3])
            # Expectation values of PauliZ on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(4)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Prepare 2×2 patches from the input image
        bsz = x.shape[0]
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                # Quantum feature extraction
                out = self.circuit(patch, self.params)
                patches.append(out)
        # Concatenate all quantum feature maps
        features = torch.cat(patches, dim=1)
        # Linear classification head
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQuanvolution"]
