import pennylane as qml
import numpy as np
import torch

class SelfAttention:
    """
    Quantum self‑attention block implemented with a variational circuit.
    Parameters
    ----------
    n_qubits : int
        Number of qubits used in the circuit (must be >= 1).
    device : str, optional
        Pennylane device name (default "default.qubit").
    Methods
    -------
    run(rotation_params, entangle_params, inputs, shots)
        Executes the circuit and returns a classical attention output.
    """
    def __init__(self, n_qubits: int, device: str = "default.qubit"):
        self.n_qubits = n_qubits
        self.dev = qml.device(device, wires=n_qubits)
        # Variational circuit producing expectation values of Pauli‑Z
        @qml.qnode(self.dev, interface="torch")
        def circuit(rot_params, ent_params, in_vec):
            # Encode the input vector via Y‑rotations
            for i in range(n_qubits):
                qml.RY(in_vec[i], wires=i)
            # Apply single‑qubit Z rotations (rotation_params)
            for i in range(n_qubits):
                qml.RZ(rot_params[i], wires=i)
            # Entangle adjacent qubits with controlled‑RZ gates
            for i in range(n_qubits - 1):
                qml.CRX(ent_params[i], wires=[i, i + 1])
            # Return expectation values of Z for each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        self.circuit = circuit

    def run(self, rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray,
            shots: int = 1024) -> np.ndarray:
        # Convert inputs to torch tensors
        rot = torch.tensor(rotation_params, dtype=torch.float32)
        ent = torch.tensor(entangle_params, dtype=torch.float32)
        inp = torch.tensor(inputs, dtype=torch.float32)

        # Execute the circuit to obtain expectation values
        expvals = self.circuit(rot, ent, inp)
        expvals_np = expvals.detach().numpy()

        # Treat the expectation vector as both query and key for a toy attention
        query = expvals_np
        key   = expvals_np
        scores = np.exp(query @ key.T / np.sqrt(self.n_qubits))
        scores /= scores.sum(axis=-1, keepdims=True)

        value = inputs
        return scores @ value

__all__ = ["SelfAttention"]
