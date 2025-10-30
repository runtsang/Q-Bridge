import numpy as np
import qiskit
from qiskit import Aer, execute
from qiskit.circuit.random import random_circuit


class HybridConv:
    """
    Parameterised quantum filter that emulates the behaviour of a
    classical convolution kernel.  The input values are encoded as
    rotation angles on a grid of qubits; a short randomised circuit
    followed by a few fixed rotations yields a richer feature map.
    The expectation value of measuring Pauli‑Z on all qubits is returned
    as a scalar in the range [0, 1].

    Parameters
    ----------
    kernel_size : int
        Size of the convolution kernel (square grid).
    threshold : float
        Binary threshold used to decide whether a rotation
        angle is set to 0 or π.
    shots : int
        Number of shots to run the quantum simulation.
    backend : qiskit.providers.Backend, optional
        Quantum backend; defaults to Aer qasm simulator.
    """

    def __init__(
        self,
        kernel_size: int,
        threshold: float,
        shots: int,
        backend=None,
    ):
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.n_qubits = kernel_size ** 2
        self.circuit = self._build_circuit()

    def _build_circuit(self):
        qc = qiskit.QuantumCircuit(self.n_qubits)
        # Parameterised rotations
        self.params = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i, p in enumerate(self.params):
            qc.rx(p, i)
        qc.barrier()
        # Randomised layer to increase expressivity
        qc += random_circuit(self.n_qubits, 2)
        qc.measure_all()
        return qc

    def run(self, data: np.ndarray, device=None) -> float:
        """
        Execute the circuit for a single 2‑D input array.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size) with values
            in the range [0, 255].
        device : torch.device, optional
            Device of the original PyTorch tensor (ignored in this
            pure‑quantum implementation but kept for API compatibility).

        Returns
        -------
        float
            Normalised expectation value of measuring |1> across all qubits.
        """
        flattened = data.reshape(1, self.n_qubits)

        # Bind parameters according to threshold
        param_binds = []
        for sample in flattened:
            bind = {}
            for i, val in enumerate(sample):
                bind[self.params[i]] = np.pi if val > self.threshold else 0.0
            param_binds.append(bind)

        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self.circuit)

        # Compute average number of |1> outcomes
        total_ones = 0
        for bitstring, count in result.items():
            total_ones += sum(int(bit) for bit in bitstring) * count

        return total_ones / (self.shots * self.n_qubits)

__all__ = ["HybridConv"]
