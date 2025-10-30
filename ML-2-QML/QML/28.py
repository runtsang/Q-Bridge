import numpy as np
import qiskit
from qiskit import Aer, execute
from qiskit.circuit import Parameter
from typing import Iterable

class QuantumFullyConnectedLayer:
    """
    Parameterised quantum circuit that implements a single‑qubit fully‑connected
    layer.  The circuit is equipped with a parameter‑shift gradient routine
    and a very light training loop so that it can be used interchangeably
    with the classical counterpart.
    """
    def __init__(self, n_qubits: int = 1, shots: int = 2000):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")

        self.theta = Parameter("θ")
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self._circuit.h(range(n_qubits))
        self._circuit.ry(self.theta, range(n_qubits))
        self._circuit.barrier()
        self._circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit for each theta value and return the expectation
        value of the computational basis measurement interpreted as a real
        number between 0 and 1.
        """
        results = []
        for theta in thetas:
            job = execute(self._circuit.bind_parameters({self.theta: theta}),
                          self.backend, shots=self.shots)
            counts = job.result().get_counts(self._circuit)
            probs = np.array([v / self.shots for v in counts.values()])
            states = np.array([int(k, 2) for k in counts.keys()], dtype=float)
            results.append(np.sum(states * probs))
        return np.array(results)

    def parameter_shift_gradient(self, theta: float, shift: float = np.pi/2) -> float:
        """
        Compute the gradient of the circuit expectation value with respect to
        a single parameter using the parameter‑shift rule.
        """
        pos = self.run([theta + shift])[0]
        neg = self.run([theta - shift])[0]
        return (pos - neg) / (2 * np.sin(shift))

    def train_step(self, theta: float, target: float, lr: float = 0.01) -> tuple[float, float]:
        """
        Simple gradient‑descent step on a single parameter.
        """
        grad = self.parameter_shift_gradient(theta)
        theta_new = theta - lr * grad
        loss = (theta_new - target) ** 2
        return theta_new, loss

def FCL() -> QuantumFullyConnectedLayer:
    """
    Factory function mirroring the original seed interface.
    """
    return QuantumFullyConnectedLayer()
