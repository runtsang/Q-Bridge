import pennylane as qml
import pennylane.numpy as np

class SamplerQNN:
    """
    Quantum sampler with a 3‑qubit variational circuit.
    The circuit consists of:
        • parameterised RY rotations for inputs and weights,
        • a layer of CNOTs to entangle the qubits,
        • a second layer of RY rotations for weights.
    The class exposes a `forward` method returning a probability distribution
    that can be used in hybrid training loops.
    """
    def __init__(self,
                 num_qubits: int = 3,
                 shots: int = 1024,
                 device_name: str = "default.qubit") -> None:
        self.device = qml.device(device_name, wires=num_qubits, shots=shots)

        # Number of parameters:
        # 2 for inputs (one per qubit), 4 for first weight layer,
        # 4 for second weight layer, plus entanglement.
        self.input_params = 2
        self.weight_params = 8

        @qml.qnode(self.device, interface="numpy")
        def circuit(inputs, weights):
            # Input encoding
            for i in range(self.input_params):
                qml.RY(inputs[i], wires=i)
            # First weight layer
            for i in range(self.input_params, self.input_params + 4):
                qml.RY(weights[i - self.input_params], wires=i % num_qubits)
            # Entanglement
            for i in range(num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Second weight layer
            for i in range(4):
                qml.RY(weights[4 + i], wires=i % num_qubits)
            return qml.probs(wires=None)

        self.circuit = circuit

    def forward(self,
                inputs: np.ndarray,
                weights: np.ndarray) -> np.ndarray:
        """
        Execute the circuit and return the probability distribution.
        Parameters:
            inputs  : shape (batch, 2) – two input angles.
            weights : shape (batch, 8) – eight variational angles.
        Returns:
            probs   : shape (batch, 2^num_qubits)
        """
        # Ensure inputs and weights are numpy arrays
        inputs = np.atleast_2d(inputs)
        weights = np.atleast_2d(weights)
        probs = []
        for inp, w in zip(inputs, weights):
            probs.append(self.circuit(inp, w))
        return np.array(probs)

    # ------------------------------------------------------------------
    # Helper for hybrid training
    # ------------------------------------------------------------------
    def loss_fn(self, preds: np.ndarray, targets: np.ndarray) -> float:
        """
        Negative log‑likelihood loss for the target class.
        """
        eps = 1e-12
        probs = preds[np.arange(len(targets)), targets]
        return -np.mean(np.log(probs + eps))

    def train_step(self,
                   opt: qml.GradientDescentOptimizer,
                   inputs: np.ndarray,
                   weights: np.ndarray,
                   targets: np.ndarray) -> float:
        """
        Perform one gradient‑descent update on the variational parameters.
        """
        def cost(weights_flat):
            w = weights_flat.reshape((len(inputs), 8))
            preds = self.forward(inputs, w)
            return self.loss_fn(preds, targets)

        opt.step(cost, weights.reshape(-1))
        return cost(weights.reshape(-1))

__all__ = ["SamplerQNN"]
