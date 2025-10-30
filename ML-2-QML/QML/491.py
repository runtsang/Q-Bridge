import pennylane as qml
import numpy as np
from pennylane import numpy as pnp

class ConvGen128QML:
    """Variational quantum filter that mimics a 128×128 kernel.

    The circuit maps each pixel to a qubit rotation and then applies a
    shallow entangling ansatz.  The expectation value of PauliZ on
    the first qubit is returned as a scalar.  The circuit is
    differentiable with respect to its parameters and can be
    optimized with any classical optimiser.
    """
    def __init__(self,
                 kernel_size: int = 128,
                 num_qubits: int | None = None,
                 device: str = 'default.qubit',
                 shots: int = 1024,
                 threshold: float = 0.5):
        self.kernel_size = kernel_size
        self.num_qubits = num_qubits or min(64, kernel_size**2)
        self.threshold = threshold
        self.backend = qml.device(device, wires=self.num_qubits, shots=shots)
        self.params = pnp.random.uniform(0, 2*np.pi, size=self.num_qubits)
        self._build_qnode()

    def _ansatz(self, params, x):
        # Encode data: rotate each qubit conditioned on pixel value
        for i in range(self.num_qubits):
            if x[i] > self.threshold:
                qml.RX(np.pi, wires=i)
            else:
                qml.RX(0.0, wires=i)
        # Variational layers
        for i in range(self.num_qubits):
            qml.RZ(params[i], wires=i)
            qml.Hadamard(wires=i)
            # Entangle nearest neighbours
            qml.CNOT(wires=[i, (i+1)%self.num_qubits])

    def _build_qnode(self):
        @qml.qnode(self.backend, interface='autograd')
        def circuit(params, x):
            self._ansatz(params, x)
            return qml.expval(qml.PauliZ(0))
        self.circuit = circuit

    def run(self, data):
        """Execute the circuit on flattened data.

        Parameters
        ----------
        data : array-like of shape (kernel_size, kernel_size)
            Input image.

        Returns
        -------
        float
            Expectation value returned by the circuit.
        """
        flat = np.reshape(data, -1)
        if flat.size > self.num_qubits:
            flat = flat[:self.num_qubits]
        elif flat.size < self.num_qubits:
            flat = np.pad(flat, (0, self.num_qubits - flat.size))
        return float(self.circuit(self.params, flat))

    def train(self,
              data,
              target,
              lr: float = 0.01,
              epochs: int = 100):
        """Simple gradient‑descent optimiser for the circuit parameters."""
        opt = qml.AdamOptimizer(stepsize=lr)
        loss_fn = lambda pred, tgt: (pred - tgt)**2
        for _ in range(epochs):
            loss, grads = opt.step_and_cost(self.circuit, self.params, data, target)
            self.params = opt.apply_gradients(grads, self.params)
        return loss

def ConvGen128QMLFactory(**kwargs):
    """Factory that returns a ConvGen128QML instance."""
    return ConvGen128QML(**kwargs)

__all__ = ["ConvGen128QML", "ConvGen128QMLFactory"]
