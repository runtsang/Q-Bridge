"""Quantum neural network regressor with a 2‑qubit variational circuit."""

import pennylane as qml
import numpy as np
import torch

dev = qml.device("default.qubit", wires=2)

def _variational_circuit(params, input_features):
    # Encode input features into rotations on the first qubit
    qml.RY(input_features[0], wires=0)
    qml.RZ(input_features[1], wires=0)
    # Parameterized rotations
    qml.RY(params[0], wires=0)
    qml.RZ(params[1], wires=0)
    qml.RY(params[2], wires=1)
    qml.RZ(params[3], wires=1)
    # Entanglement
    qml.CNOT(wires=[0, 1])
    # Further rotations
    qml.RY(params[4], wires=0)
    qml.RZ(params[5], wires=0)
    qml.RY(params[6], wires=1)
    qml.RZ(params[7], wires=1)
    # Measurement: expectation of Z on first qubit
    return qml.expval(qml.Z(0))

class EstimatorQNNExtended:
    """A quantum estimator that extends the original classical network."""
    def __init__(self, weight_params: np.ndarray | None = None) -> None:
        # 8 trainable parameters
        self.num_params = 8
        if weight_params is None:
            # Random initialization
            self.weight_params = np.random.randn(self.num_params)
        else:
            self.weight_params = weight_params

        # Create a QNode with autograd interface
        self.qnode = qml.QNode(_variational_circuit, dev, interface="autograd")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Evaluate the variational circuit for each input sample."""
        # Convert torch tensor to numpy for Pennylane
        inputs_np = inputs.detach().cpu().numpy()
        preds_np = []
        for sample in inputs_np:
            pred = self.qnode(self.weight_params, sample)
            preds_np.append(pred)
        preds_np = np.array(preds_np)
        return torch.from_numpy(preds_np)

__all__ = ["EstimatorQNNExtended"]
