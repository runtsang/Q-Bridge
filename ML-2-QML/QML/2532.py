import numpy as np
import qiskit
from typing import Iterable

class HybridQuantumCircuit:
    """
    Quantum analogue of HybridFCLQuanvolution.
    Uses a 5‑qubit circuit: four qubits encode a 2×2 patch via Ry rotations
    (mimicking the quanvolution kernel) and the fifth qubit implements
    a parameterised fully connected layer with a single Ry rotation.
    """

    def __init__(self, backend: qiskit.providers.Backend, shots: int = 1024) -> None:
        self.backend = backend
        self.shots = shots
        self._circuit = qiskit.QuantumCircuit(5)
        # Parameters
        self.theta_fc = qiskit.circuit.Parameter("theta_fc")
        self.theta_conv = qiskit.circuit.ParameterVector("theta_conv", 4)
        # Build circuit
        self._circuit.h(range(5))
        # Fully connected rotation on qubit 4
        self._circuit.ry(self.theta_fc, 4)
        # Convolutional rotations on qubits 0‑3
        for i, p in enumerate(self.theta_conv):
            self._circuit.ry(p, i)
        self._circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit with a list of 5 parameters:
        [theta_fc, theta_conv0, theta_conv1, theta_conv2, theta_conv3].
        Returns the expectation value of the measurement outcome distribution.
        """
        if len(thetas)!= 5:
            raise ValueError("Expected 5 parameters: one for FC and four for convolution.")
        param_bind = {self.theta_fc: thetas[0]}
        for i, p in enumerate(self.theta_conv):
            param_bind[p] = thetas[i + 1]
        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[param_bind],
        )
        result = job.result().get_counts(self._circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probabilities = counts / self.shots
        expectation = np.sum(states * probabilities)
        return np.array([expectation])

__all__ = ["HybridQuantumCircuit"]
