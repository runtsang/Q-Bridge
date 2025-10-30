# EstimatorQNN__gen279.py – Quantum‑ML extension (PennyLane)

"""
A two‑qubit variational quantum circuit that mirrors the original single‑qubit architecture
but adds expressive depth and integrates with PyTorch via PennyLane’s TorchLayer.
The circuit is trained to perform regression on a 2‑dimensional input vector.
"""

import pennylane as qml
import torch
from torch import nn
from torch.utils.data import DataLoader


class EstimatorQNN(nn.Module):
    """
    Variational quantum neural network.

    Parameters
    ----------
    n_qubits : int, default 2
        Number of qubits in the ansatz.
    layers : int, default 2
        Depth of the hardware‑efficient circuit.
    device : str, default "default.qubit"
        Pennylane device name (e.g., "default.qubit").
    """

    def __init__(self, n_qubits: int = 2, layers: int = 2, device: str = "default.qubit") -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.layers = layers
        dev = qml.device(device, wires=n_qubits)

        def circuit(inputs: tuple[float,...], weights: torch.Tensor) -> float:
            # Input encoding (angle‑encoding)
            for i, w in enumerate(inputs):
                qml.RY(w, wires=i)
            # Parameterized ansatz
            for _ in range(layers):
                for i in range(n_qubits):
                    qml.RX(weights[0, i], wires=i)
                    qml.RZ(weights[1, i], wires=i)
                # Entangling layer
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            # Measurement: expectation of Z on first qubit
            return qml.expval(qml.PauliZ(0))

        self.qnode = qml.QNode(circuit, dev, interface="torch")
        self.torch_layer = qml.qnn.TorchLayer(self.qnode, weight_shapes={"weights": (2, n_qubits)})

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape (..., 2). Must contain two input features.
        """
        # Ensure inputs are float32 and detached from any graph
        inp = inputs.detach().float()
        return self.torch_layer(inp)

    # ------------------------------------------------------------------
    # Convenience training helper
    # ------------------------------------------------------------------
    def train_model(
        self,
        loader: DataLoader,
        loss_fn: nn.Module = nn.MSELoss(),
        optimizer: torch.optim.Optimizer | None = None,
        epochs: int = 20,
        device: torch.device | str = "cpu",
    ) -> list[float]:
        """
        Trains the quantum circuit on the supplied DataLoader.

        Returns
        -------
        losses : list[float]
            Training loss per epoch.
        """
        self.to(device)
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        losses: list[float] = []
        for _ in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = self(xb).squeeze()
                loss = loss_fn(pred, yb.squeeze())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            losses.append(epoch_loss / len(loader.dataset))
        return losses


__all__ = ["EstimatorQNN"]
