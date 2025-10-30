import pennylane as qml
import pennylane.numpy as np

class FCL:
    """
    Quantum fully connected layer implemented with PennyLane.
    Encodes input as a rotation angle and returns the expectation value of PauliZ.
    """

    def __init__(self, n_wires: int = 1, dev_name: str = "default.qubit"):
        self.n_wires = n_wires
        self.dev = qml.device(dev_name, wires=n_wires)
        # Variational parameters
        self.theta = np.random.randn(1)

        # Define the variational circuit as a QNode
        def _circuit(x, params):
            qml.Hadamard(wires=range(self.n_wires))
            qml.RY(params[0] * x, wires=range(self.n_wires))
            for i in range(self.n_wires - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

        self.circuit = qml.QNode(_circuit, self.dev)

    def run(self, thetas):
        """
        Evaluate the circuit expectation value for each theta.
        """
        expectations = [self.circuit(theta, [theta]) for theta in thetas]
        return np.array(expectations)

    def optimize(self, data_loader, epochs: int = 100, lr: float = 0.01):
        """
        Jointly optimize the circuit parameters on a dataset.
        """
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        for epoch in range(epochs):
            for batch_x, batch_y in data_loader:
                def cost(params):
                    preds = [self.circuit(x, params) for x in batch_x]
                    preds = np.array(preds)
                    return np.mean((preds - batch_y.numpy()) ** 2)
                self.theta = opt.step(cost, self.theta)
