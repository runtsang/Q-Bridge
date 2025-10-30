import pennylane as qml
import numpy as np

class QCNN:
    """
    A hybrid QCNN implemented with Pennylane.
    Architecture:
        • 8‑qubit Z‑feature map
        • Three convolution‑pooling stages
        • Measurement of the first qubit in the Z basis
    The circuit is fully differentiable via the parameter‑shift rule
    and can be trained with an Adam optimiser.
    """
    def __init__(self, dev: str = "default.qubit", seed: int = 1234):
        self.dev = qml.device(dev, wires=8, shots=1000)
        self.rng = np.random.default_rng(seed)
        self._init_params()
        self._build_circuit()

    def _init_params(self):
        # 3 parameters per 2‑qubit gate, 4 gates per layer
        self.conv_params = [self.rng.standard_normal(3) for _ in range(4)]
        self.pool_params = [self.rng.standard_normal(3) for _ in range(4)]

    def _conv_block(self, wires, params):
        """Two‑qubit convolution gate."""
        qml.RZ(-np.pi / 2, wires=wires[1])
        qml.CNOT(wires[1], wires[0])
        qml.RZ(params[0], wires=wires[0])
        qml.RY(params[1], wires=wires[1])
        qml.CNOT(wires[0], wires[1])
        qml.RY(params[2], wires=1)
        qml.CNOT(wires[1], wires[0])
        qml.RZ(np.pi / 2, wires=0)

    def _pool_block(self, wires, params):
        """Two‑qubit pooling gate."""
        qml.RZ(-np.pi / 2, wires=wires[1])
        qml.CNOT(wires[1], wires[0])
        qml.RZ(params[0], wires=wires[0])
        qml.RY(params[1], wires=wires[1])
        qml.CNOT(wires[0], wires[1])
        qml.RY(params[2], wires=1)

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(x):
            # Feature map
            for i, val in enumerate(x):
                qml.RZ(val, wires=i)
                qml.RX(val, wires=i)

            # Stage 1
            for j in range(0, 8, 2):
                self._conv_block([j, j+1], self.conv_params[0])
            for j in range(0, 8, 2):
                self._pool_block([j, j+1], self.pool_params[0])

            # Stage 2
            for j in range(0, 4, 2):
                self._conv_block([j+4, j+5], self.conv_params[1])
            for j in range(0, 4, 2):
                self._pool_block([j+4, j+5], self.pool_params[1])

            # Stage 3
            for j in range(0, 2, 2):
                self._conv_block([j+6, j+7], self.conv_params[2])
            for j in range(0, 2, 2):
                self._pool_block([j+6, j+7], self.pool_params[2])

            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return probabilities for an array of input feature vectors."""
        preds = np.array([self.circuit(x) for x in X])
        return 1 / (1 + np.exp(-preds))

    def _loss(self, X, y):
        preds = self.predict(X)
        return np.mean((preds - y) ** 2)

    def _gradients(self, X, y):
        """Compute gradients of the loss w.r.t. all parameters."""
        grads = {}
        # Gradient w.r.t. convolution params
        grads['conv'] = []
        for i in range(len(self.conv_params)):
            grad = np.zeros_like(self.conv_params[i])
            for j in range(3):
                param = self.conv_params[i][j]
                # parameter‑shift rule
                shift = np.pi / 2
                self.conv_params[i][j] = param + shift
                f_plus = self._loss(X, y)
                self.conv_params[i][j] = param - shift
                f_minus = self._loss(X, y)
                self.conv_params[i][j] = param
                grad[j] = (f_plus - f_minus) / (2 * shift)
            grads['conv'].append(grad)
        # Gradient w.r.t. pooling params
        grads['pool'] = []
        for i in range(len(self.pool_params)):
            grad = np.zeros_like(self.pool_params[i])
            for j in range(3):
                param = self.pool_params[i][j]
                shift = np.pi / 2
                self.pool_params[i][j] = param + shift
                f_plus = self._loss(X, y)
                self.pool_params[i][j] = param - shift
                f_minus = self._loss(X, y)
                self.pool_params[i][j] = param
                grad[j] = (f_plus - f_minus) / (2 * shift)
            grads['pool'].append(grad)
        return grads

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 200, lr: float = 0.01, verbose: bool = False):
        """Simple Adam optimizer for the QCNN parameters."""
        opt = qml.AdamOptimizer(stepsize=lr)
        params = (self.conv_params, self.pool_params)

        for epoch in range(epochs):
            # Flatten parameters for optimizer
            flat_params = [p for grp in params for p in grp]
            flat_params, loss = opt.step_and_cost(
                lambda p: self._loss(X, y), flat_params
            )
            # Re‑bundle parameters
            self.conv_params = [flat_params[i*3:(i+1)*3] for i in range(4)]
            self.pool_params = [flat_params[12 + i*3:12 + (i+1)*3] for i in range(4)]

            if verbose and epoch % 20 == 0:
                print(f"Epoch {epoch:3d} Loss: {loss:.4f}")

__all__ = ["QCNN"]
