import pennylane as qml
import pennylane.numpy as np

def SamplerQNN(num_qubits: int = 2,
               entanglement: str = "full",
               device: str = "default.qubit") -> qml.QNode:
    """Quantum sampler network using Pennylane.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the variational circuit.
    entanglement : str
        Entanglement strategy for the Hadamard layers ("full", "circular", "linear").
    device : str
        Pennylane device name.

    Returns
    -------
    qml.QNode
        Variational circuit that outputs a probability distribution over
        basis states. The circuit accepts two arguments:
        * inputs: array of size `num_qubits` for Ry encoding.
        * weights: array of shape `(num_layers, num_qubits)` for Ry rotations.
    """
    dev = qml.device(device, wires=num_qubits)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        # Input encoding: Ry rotations
        for i in range(num_qubits):
            qml.RY(inputs[i], wires=i)
        # Variational layers
        for w in weights:
            for i in range(num_qubits):
                qml.RY(w[i], wires=i)
            qml.layer(qml.CNOT, wires=range(num_qubits), entanglement=entanglement)
        # Measurement: probabilities of all basis states
        return qml.probs(wires=range(num_qubits))

    return circuit

__all__ = ["SamplerQNN"]
