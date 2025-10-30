import pennylane as qml
import torch

def make_qcnn_qnode(device, num_qubits=8, num_layers=3):
    """Create a variational quantum circuit that mirrors the original QCNN layout."""
    dev = qml.device(device, wires=num_qubits)

    @qml.qnode(dev, interface="torch")
    def qnode(inputs: torch.Tensor, weights: torch.Tensor):
        # Feature map: encode inputs as Z rotations
        for i, val in enumerate(inputs):
            qml.RZ(val, wires=i)

        # Ansatz layers
        for layer in range(num_layers):
            for q1 in range(0, num_qubits, 2):
                qml.CNOT(q1, q1+1)
                # 3 parameter rotations for each two‑qubit block
                qml.RX(weights[layer, q1//2, 0], wires=q1)
                qml.RY(weights[layer, q1//2, 1], wires=q1+1)
                qml.RZ(weights[layer, q1//2, 2], wires=q1+1)

        # Measurement: expectation of Z on all qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    return qnode

__all__ = ["make_qcnn_qnode"]
