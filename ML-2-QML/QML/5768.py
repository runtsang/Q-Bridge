import torch
import torch.nn as nn
import pennylane as qml


class QuantumNATExtended(nn.Module):
    """Quantum model with a tunable, parameter‑efficient ansatz and simple feature encoding."""

    def __init__(self, num_features: int = 4, num_layers: int = 2,
                 entanglement_depth: int = 1) -> None:
        """
        Parameters
        ----------
        num_features : int
            Number of qubits / output dimensions.
        num_layers : int
            Depth of the variational ansatz.
        entanglement_depth : int
            Distance of the CNOT entanglement in each layer.
        """
        super().__init__()
        self.num_features = num_features
        self.num_layers = num_layers
        self.entanglement_depth = entanglement_depth

        self.norm = nn.BatchNorm1d(num_features)

        # Device for the variational circuit
        self.device = qml.device("default.qubit", wires=num_features)

        # Create a QNode that encodes data with RY rotations and applies a
        # simple, parameter‑efficient ansatz with tunable entanglement.
        def circuit(params, data):
            # Data encoding
            for i in range(num_features):
                qml.RY(data[i], wires=i)

            # Variational ansatz
            for layer in range(num_layers):
                for i in range(num_features):
                    qml.Rot(
                        params[layer, i, 0],
                        params[layer, i, 1],
                        params[layer, i, 2],
                        wires=i,
                    )
                # Entangling layer
                for i in range(num_features - entanglement_depth):
                    qml.CNOT(wires=[i, i + entanglement_depth])

            return [qml.expval(qml.PauliZ(i)) for i in range(num_features)]

        self.qnode = qml.QNode(circuit, self.device, interface="torch")

        # Simple linear encoder to map arbitrary inputs to the qubit dimension
        self.encoder = nn.Linear(1, num_features)

    def feature_extractor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project the input tensor to the qubit dimension.
        The input is expected to be of shape (batch, *).
        """
        # Collapse all but the batch dimension to a single scalar per sample
        flat = x.view(x.shape[0], -1)
        scalar = flat.mean(dim=1, keepdim=True)
        return self.encoder(scalar)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        batch_size = x.shape[0]
        features = self.feature_extractor(x)  # shape (batch, num_features)

        # Random parameters for the ansatz
        params = torch.randn(
            self.num_layers, self.num_features, 3, device=x.device
        )

        # Evaluate the circuit for each sample in the batch
        outputs = []
        for feat in features:
            out = self.qnode(params, feat)
            outputs.append(out)
        out_tensor = torch.stack(outputs)  # (batch, num_features)

        return self.norm(out_tensor)


__all__ = ["QuantumNATExtended"]
