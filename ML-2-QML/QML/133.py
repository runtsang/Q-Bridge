import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter

class FCL:
    """
    Parameterised quantum circuit that emulates a fully‑connected layer.

    The circuit consists of a Hadamard layer followed by a single
    Ry rotation that is parameterised by ``theta``.  The `run` method
    evaluates the expectation value of the Pauli‑Z operator on the
    first qubit for each supplied parameter value.  A simple
    parameter‑shift gradient estimator is provided together with a
    tiny learning‑rate optimiser that can be used in a gradient‑based
    training loop.
    """

    def __init__(self, n_qubits: int = 1, shots: int = 1000):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        self.theta = Parameter("theta")
        self.circuit = QuantumCircuit(n_qubits)
        # Prepare an equal‑weight superposition
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        # Parameterised Ry gate on all qubits
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Evaluate the expectation value of Z on the first qubit for each
        parameter value supplied in ``thetas``.
        """
        expectations = []
        for theta in thetas:
            job = execute(self.circuit,
                          backend=self.backend,
                          shots=self.shots,
                          parameter_binds=[{self.theta: theta}])
            result = job.result()
            counts = result.get_counts(self.circuit)
            exp_val = 0.0
            for outcome, count in counts.items():
                # outcome is a bitstring; first qubit is the leftmost bit
                z = 1 if outcome[0] == "0" else -1
                exp_val += z * count
            exp_val /= self.shots
            expectations.append(exp_val)
        return np.array(expectations)

    def gradient(self, thetas: np.ndarray) -> np.ndarray:
        """
        Estimate the gradient of the expectation value w.r.t. each
        parameter using the parameter‑shift rule.
        """
        shift = np.pi / 2
        grads = []
        for theta in thetas:
            f_plus = self.run(np.array([theta + shift]))[0]
            f_minus = self.run(np.array([theta - shift]))[0]
            grads.append((f_plus - f_minus) / 2.0)
        return np.array(grads)

    def train_step(self, thetas: np.ndarray, target: float, lr: float = 0.01) -> tuple[float, np.ndarray]:
        """
        Perform one gradient‑descent step.

        Parameters
        ----------
        thetas : np.ndarray
            Current parameters.
        target : float
            Desired expectation value.
        lr : float
            Learning rate.

        Returns
        -------
        loss : float
            Squared‑error loss after the update.
        updated_thetas : np.ndarray
            Parameters after the update.
        """
        preds = self.run(thetas)
        loss = np.mean((preds - target) ** 2)
        grads = self.gradient(thetas)
        updated = thetas - lr * grads
        return loss, updated
