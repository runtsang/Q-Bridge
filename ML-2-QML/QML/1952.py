import pennylane as qml
import numpy as np
import torch
from torch import nn

class QCNNHybrid(nn.Module):
    """
    Hybrid quantum‑classical QCNN implemented with PennyLane.
    The network is structured as:
        1. Classical feature map (ZFeatureMap).
        2. Three convolution‑pooling blocks implemented as
           parameterised two‑qubit unitaries.
        3. Final measurement of the first qubit in the Z basis.
    The class exposes a forward method that accepts a Torch tensor,
    converts it to a NumPy array, evaluates the hybrid circuit
    and returns a probability in (0, 1).
    """

    def __init__(self, input_dim: int = 8, device: str = "default.qubit") -> None:
        super().__init__()
        self.input_dim = input_dim
        self.device = device

        # PennyLane device
        self.dev = qml.device(self.device, wires=self.input_dim)

        # Feature map
        self.feature_map = qml.templates.feature_maps.ZFeatureMap(
            wires=range(self.input_dim), depth=2, entanglement="full"
        )

        # Ansatz layers
        self.ansatz_layers = nn.ModuleList()
        for layer_idx in range(3):
            self.ansatz_layers.append(self._conv_pool_layer(layer_idx))

        # Trainable parameters
        self.params = nn.ParameterList(
            [nn.Parameter(torch.randn(3 * (2 ** l))) for l in range(3)]
        )

        # Optimiser placeholder (user can set later)
        self.optimizer = None

    def _conv_pool_layer(self, idx: int):
        """Creates a two‑qubit convolution‑pooling block."""
        def layer(params, wires):
            # params: (3,) for each pair of qubits
            for i in range(0, len(wires), 2):
                q1, q2 = wires[i], wires[i + 1]
                qml.RZ(-np.pi / 2, wires=q2)
                qml.CNOT(q2, q1)
                qml.RZ(params[0], wires=q1)
                qml.RY(params[1], wires=q2)
                qml.CNOT(q1, q2)
                qml.RY(params[2], wires=q2)
                qml.CNOT(q2, q1)
                qml.RZ(np.pi / 2, wires=q1)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (batch, input_dim)
            Input data for the QNN.

        Returns
        -------
        torch.Tensor of shape (batch, 1)
            Probabilities after evaluating the hybrid circuit.
        """
        def qnode_fn(inputs, *weights):
            # Encode classical features
            self.feature_map(inputs)

            # Apply ansatz layers
            for layer, w in zip(self.ansatz_layers, weights):
                layer(w, range(self.input_dim))

            # Measurement
            return qml.expval(qml.PauliZ(0))

        qnode = qml.QNode(qnode_fn, self.dev, interface="torch")

        # Prepare weights list
        flat_weights = [p for p in self.params]
        out = qnode(x, *flat_weights)
        # Convert to probability
        return torch.sigmoid(out.unsqueeze(-1))

def QCNNHybridFactory() -> QCNNHybrid:
    """
    Factory returning a default configuration of the PennyLane QCNNHybrid.
    """
    return QCNNHybrid()

__all__ = ["QCNNHybrid", "QCNNHybridFactory"]
