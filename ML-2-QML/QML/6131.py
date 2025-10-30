import pennylane as qml  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import numpy as np  

class QuanvolutionNet(nn.Module):  
    """Quantum variant of the Quanvolution model.  
    Applies a variational quantum circuit to each 2×2 image patch.  
    Uses PennyLane with a default qubit device."""  

    def __init__(self, dev_name="default.qubit", wires=4, n_layers=2, shots=1024):  
        super().__init__()  
        self.wires = wires  
        self.n_layers = n_layers  
        self.dev = qml.device(dev_name, wires=wires, shots=shots)  
        # Trainable parameters for the variational layers  
        self.variational_params = nn.Parameter(torch.randn(n_layers, wires))  
        self.linear = nn.Linear(4 * 14 * 14, 10)  

    def _qnode(self, patch: torch.Tensor, params: torch.Tensor):  
        @qml.qnode(self.dev, interface="torch")  
        def circuit(patch, params):  
            # Encode patch values into RY rotations  
            for i in range(self.wires):  
                qml.RY(patch[:, i], wires=i)  
            # Variational layers  
            for l in range(self.n_layers):  
                for i in range(self.wires):  
                    qml.RZ(params[l, i], wires=i)  
                for i in range(self.wires - 1):  
                    qml.CNOT(wires=[i, i + 1])  
            # Measure in the Z basis  
            return [qml.expval(qml.PauliZ(i)) for i in range(self.wires)]  
        return circuit(patch, params)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        # x: (batch, 1, 28, 28)  
        batch_size = x.size(0)  
        patches = []  
        x = x.view(batch_size, 28, 28)  
        for r in range(0, 28, 2):  
            for c in range(0, 28, 2):  
                # Extract 2×2 patch  
                patch = x[:, r:r+2, c:c+2]  # (batch, 2, 2)  
                patch = patch.reshape(batch_size, -1)  # (batch, 4)  
                # Normalize to [-π, π] for RY rotations  
                patch = (patch - patch.mean()) / (patch.std() + 1e-6) * np.pi  
                measurement = self._qnode(patch, self.variational_params)  # (batch, 4)  
                patches.append(measurement)  
        features = torch.cat(patches, dim=1)  # (batch, 4*14*14)  
        logits = self.linear(features)  
        return F.log_softmax(logits, dim=-1)  

__all__ = ["QuanvolutionNet"]
