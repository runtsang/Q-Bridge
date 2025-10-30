import pennylane as qml
import torch

class SelfAttention:
    """
    Variational self‑attention implemented with Pennylane.
    The circuit uses rotation layers per qubit and controlled entangling gates,
    producing expectation values that can be used as attention weights.
    """
    def __init__(self, n_qubits: int, num_heads: int = 1, device: str = "default.qubit"):
        self.n_qubits = n_qubits
        self.num_heads = num_heads
        self.dev = qml.device(device, wires=n_qubits)
        self._build_qnode()

    def _build_qnode(self):
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(params: torch.Tensor, entangle: torch.Tensor):
            # params shape: (num_heads, n_qubits, 3)  Euler angles
            # entangle shape: (num_heads, n_qubits-1)
            for h in range(self.num_heads):
                for q in range(self.n_qubits):
                    theta_x, theta_y, theta_z = params[h, q]
                    qml.RX(theta_x, wires=q)
                    qml.RY(theta_y, wires=q)
                    qml.RZ(theta_z, wires=q)
                for q in range(self.n_qubits - 1):
                    qml.CRX(entangle[h, q], wires=[q, q + 1])
            return [qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits)]

        self.circuit = circuit

    def run(self, params: torch.Tensor, entangle: torch.Tensor, shots: int = 1024) -> torch.Tensor:
        """
        Execute the variational circuit and return expectation values.

        Parameters
        ----------
        params : torch.Tensor
            Rotation parameters of shape (num_heads, n_qubits, 3).
        entangle : torch.Tensor
            Entangling parameters of shape (num_heads, n_qubits-1).
        shots : int
            Number of shots for sampling (ignored when using default.qubit).

        Returns
        -------
        torch.Tensor
            Expectation values of PauliZ for each qubit: shape (n_qubits,).
        """
        return self.circuit(params, entangle)

__all__ = ["SelfAttention"]
